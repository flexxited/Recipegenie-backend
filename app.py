import os
import uuid
import re
import aiohttp
import asyncio
import openai
from time import time
from google.cloud import firestore, secretmanager, storage
from google.oauth2 import service_account
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import json
from datetime import datetime

# Initialize Secret Manager client
secret_client = secretmanager.SecretManagerServiceClient()

# Function to access secret
def access_secret_version(secret_name):
    response = secret_client.access_secret_version(name=secret_name)
    return response.payload.data.decode('UTF-8')

# Access the OpenAI API key from Secret Manager
openai_api_key = access_secret_version("projects/the-recipe-genie/secrets/OPENAI_API_KEY/versions/latest")

# Access the service account key from Secret Manager
service_account_info = json.loads(access_secret_version("projects/the-recipe-genie/secrets/my-service-account-key/versions/latest"))
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Initialize Firestore client with the credentials
db = firestore.Client(credentials=credentials, project=os.getenv('GOOGLE_CLOUD_PROJECT'))

# Initialize OpenAI client
aclient = openai.AsyncOpenAI(api_key=openai_api_key)

# Initialize Cloud Storage client
storage_client = storage.Client(credentials=credentials)
bucket_name = "the-recipe-genie.appspot.com"    
bucket = storage_client.bucket(bucket_name)

app = Flask(__name__)
CORS(app)

RATE_LIMIT = 100
RATE_LIMIT_WINDOW = 3600

def generate_api_key():
    return str(uuid.uuid4())

@app.route('/subscribe', methods=['POST'])
def subscribe():
    data = request.json
    unique_id = data.get('unique_id')
    subscription_plan = data.get('subscription_plan')

    if not unique_id or not subscription_plan:
        return jsonify({'error': 'Unique ID and subscription plan are required'}), 400

    user_doc_ref = db.collection('users').document(unique_id)
    user_doc_ref.set({
        'subscription_plan': subscription_plan,
        'subscription_date': firestore.SERVER_TIMESTAMP,
        'isSubscriptionActive': True,
        'subscriptionStatus': 'active'
    }, merge=True)

    new_api_key = generate_api_key()
    db.collection('api_keys').document(new_api_key).set({
        'user_id': unique_id,
        'created_at': firestore.SERVER_TIMESTAMP,
        'request_count': 0,
        'last_request_time': None
    })

    return jsonify({'api_key': new_api_key})

def verify_api_key(api_key):
    doc = db.collection('api_keys').document(api_key).get()
    return doc.exists

def rate_limit(api_key):
    doc_ref = db.collection('api_keys').document(api_key)
    doc = doc_ref.get()

    if not doc.exists:
        return False

    data = doc.to_dict()
    current_time = time()
    last_request_time = data.get('last_request_time')

    if last_request_time:
        last_request_time_seconds = last_request_time.timestamp()
        elapsed_time = current_time - last_request_time_seconds
    else:
        elapsed_time = RATE_LIMIT_WINDOW + 1

    request_count = data.get('request_count', 0)

    if elapsed_time > RATE_LIMIT_WINDOW:
        doc_ref.update({
            'request_count': 1,
            'last_request_time': firestore.SERVER_TIMESTAMP
        })
    else:
        if request_count >= RATE_LIMIT:
            return False
        else:
            doc_ref.update({
                'request_count': firestore.Increment(1),
                'last_request_time': firestore.SERVER_TIMESTAMP
            })

    return True

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if not api_key or not verify_api_key(api_key):
            return jsonify({'error': 'Invalid or missing API key'}), 401
        if not rate_limit(api_key):
            return jsonify({'error': 'Rate limit exceeded'}), 429
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return "Welcome to the Recipe and Image Generator API"

def generate_prompt(ingredients, num_people, dietary_choice, allergies):
    prompt = (
        f"""You are a master chef. Each time you receive a request, treat it independently and do not consider any previous context or recipes. Now, think of the setup as a restaurant. Now the customer can select ingredients placed at the counter. Your goal here is to develop a fabulous and edible recipe using the ingredients the customer selects. You may include all ingredients selected by the customer or exclude ingredients if they aren't necessary, however, it is very important for you to exclude any ingredients that the customer is allergic to and not suitable according to the selected dietary choices {', '.join(dietary_choice)} . Keep in mind that you cannot add any ingredients that the customer hasn't selected apart from basic ones such as oil, water, and salt.
        Your goal here would be to inform the customer for a dish/beverage that already exists and you know of or come up with your recipe as an alternative and it must obey the dietary choices provided by the customer. For every recipe, the format of the recipe strictly needs to be:
        **Recipe Name**
        **Ingredients**
        **Instructions**
        **Nutritional value**
        It needs to be within 500 tokens strictly.
        Don't consider the previously generated receipe with same ingredient  
        Very important note: The recipe has to be strictly using only the ingredients selected by the customer. No additional ingredients should be added by you.
        Customer 1:
        The ingredient(s) selected are {', '.join(ingredients)}.
        Number of people to cook for {num_people}
        The customer is/are allergic to {', '.join(allergies)}, 
        The customer has dietary restrictions that include: {', '.join(dietary_choice)}.
        Your challenge is to create a recipe using only the provided ingredients while adhering strictly to the dietary restrictions and avoiding all allergens.
        Ensure the recipe is suitable for the dietary choices and free from any allergens
        Provide us with a Visualization Prompt that will help us to generate and display the realistic image of the prepared recipe, plated and ready to be served.
        No introductory or summary lines."""
    )
    print("Generated prompt:", prompt)  # Debug print
    return prompt

def check_recipe_failure(recipe_text):
    failure_phrases = [
        "unable to generate",
        "cannot be reconciled",
        "impossible to create",
        "failed to create",
        "cannot generate",
        "unable to produce",
        "error generating",
        "failed to produce",
        "due to",
        "because of",
        "as a result of",
        "due to the constraints",
        "because of the restrictions",
        "insufficient ingredients",
        "lack of required ingredients",
        "unable to comply with the dietary restrictions",
        "allergy constraints",
        "I'm sorry",
        "unfortunately",
        "regrettably",
        "apologies",
        "please note",
        "note that",
        "take note",
        "consider using different ingredients",
        "try adjusting",
        "you might want to",
        "please try again",
        "check the ingredients",
        "reevaluate the constraints",
        "you can modify",
        "recipe not possible",
        "unable to proceed",
        "unable to fulfill the request",
        "cannot comply",
        "cannot accommodate",
        "recipe generation unsuccessful",
        "unable to formulate",
        "request cannot be completed",
        "unable to create",
        "unable to craft",
        "insufficient viable ingredients",
        "restrictions too limiting",
        "constraints not met",
        "does not meet criteria",
        "unable to meet requirements",
        "does not adhere to guidelines",
        "unable to satisfy the constraints",
        "cannot accommodate the given constraints",
        "recipe creation halted",
        "unable to generate a feasible recipe",
        "unable to proceed with the given inputs",
        "combination of ingredients not workable",
        "unable to generate under given restrictions",
        "recipe creation not possible with current inputs",
        "recipe cannot be constructed",
        "unable to construct a recipe",
        "recipe formulation failed",
        "it is not possible to create",
        "it is not possible to create a recipe"
        "unable to provide a suitable recipe"
    ]
    
    recipe_text_lower = recipe_text.lower()
    for phrase in failure_phrases:
        if phrase in recipe_text_lower:
            return True
    return False

async def generate_image_prompt(recipe_text):
    vis_prompt_match = re.search(r"\*\*Visualization Prompt\*\*\n(.+)", recipe_text)
    if vis_prompt_match:
        vis_prompt = vis_prompt_match.group(1).strip()
        recipe_text = recipe_text.replace(vis_prompt_match.group(0), "").strip()
        return vis_prompt, recipe_text, True
    else:
        recipe_text = re.sub(r"\*\*Visualization Prompt\*\*\n(.+)", "", recipe_text).strip()
        return "Default visualization prompt based on recipe", recipe_text, False

async def fetch_image(image_prompt):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'https://api.openai.com/v1/images/generations',
            json={
                "model": "dall-e-3",
                "prompt": image_prompt,
                "size": "1024x1024",
                "n": 1
            },
            headers={
                'Authorization': f'Bearer {openai_api_key}'
            }
        ) as response:
            if response.status != 200:
                return {'error': 'Failed to generate image'}
            data = await response.json()
            if not data.get('data'):
                return {'error': 'Failed to generate image'}
            return {'imageUrl': data['data'][0]['url']}

async def download_image(image_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status != 200:
                raise Exception("Failed to download image")
            image_data = await response.read()
    return image_data

def upload_to_cloud_storage(image_data, filename):
    blob = bucket.blob(filename)
    blob.upload_from_string(image_data, content_type='image/png')
    blob.make_public()
    return blob.public_url

def remove_nutritional_info(recipe_text):
    # Regular expression to match **Nutritional Value** section
    recipe_text = re.sub(r"\*\*Nutritional Value\*\*.*", "", recipe_text, flags=re.DOTALL).strip()
    # Regular expression to match "Nutritional Information" section if any
    recipe_text = re.sub(r"Nutritional Information.*", "", recipe_text, flags=re.DOTALL).strip()
    return recipe_text

@app.route('/generate_recipe_and_image', methods=['POST'])
@require_api_key
def generate_recipe_and_image():
    try:
        data = request.get_json()

        # Ensure ingredients are handled correctly
        ingredients = data.get('ingredients')
        if isinstance(ingredients, str):
            ingredients = ingredients.split(',')
        elif not isinstance(ingredients, list):
            return jsonify({'error': 'Ingredients should be a list or a comma-separated string'}), 400

        num_people = data.get('num_people')
        
        # Ensure dietary_choice is handled correctly
        dietary_choice = data.get('dietary', [])
        if isinstance(dietary_choice, str):
            dietary_choice = dietary_choice.split(',')
        elif not isinstance(dietary_choice, list):
            return jsonify({'error': 'Dietary choice should be a list or a comma-separated string'}), 400

        # Ensure allergies are handled correctly
        allergies = data.get('allergies', [])
        if isinstance(allergies, str):
            allergies = allergies.split(',')
        elif not isinstance(allergies, list):
            return jsonify({'error': 'Allergies should be a list or a comma-separated string'}), 400

        if not ingredients or not num_people:
            return jsonify({'error': 'Ingredients and number of people are required'}), 400

        # Define non-vegetarian ingredients
        non_vegetarian_ingredients = ["lamb chops", "beef steak", "pork chops", "chicken breast", "turkey breast", 
                              "duck breast", "veal cutlets", "pork tenderloin", "beef brisket", "lamb shank", 
                              "beef ribs", "pork ribs", "chicken thighs", "ground beef", "ground pork", 
                              "ground chicken", "ground turkey", "bacon", "ham", "sausage (various types)", 
                              "cornish hen", "rabbit", "venison", "quail", "pheasant", "bison steak", "goat meat", 
                              "frog legs", "wild boar", "ostrich meat", "chicken breast", "chicken thighs", 
                              "chicken drumsticks", "chicken wings", "chicken tenderloins", "whole chicken", 
                              "chicken quarters", "chicken legs", "chicken thighs (boneless, skinless)", 
                              "chicken breasts (bone-in, skin-on)", "ground chicken", "chicken sausage", 
                              "chicken liver", "chicken gizzards", "chicken hearts", "chicken back", "chicken neck", 
                              "chicken feet", "chicken cutlets", "chicken schnitzel", "chicken strips", "chicken nuggets", 
                              "chicken meatballs", "chicken patties", "rotisserie chicken", "fried chicken", 
                              "chicken kabobs", "chicken tenders", "chicken satay", "chicken cordon bleu", "beef burger patty", 
                              "turkey burger patty", "chicken burger patty", "pork burger patty", "lamb burger patty", 
                              "veal burger patty", "bison burger patty", "venison burger patty", "salmon burger patty", 
                              "tuna burger patty", "crab cake burger patty", "shrimp burger patty", "duck burger patty", 
                              "canned tuna", "canned salmon", "canned sardines", "canned anchovies", "canned clams", 
                              "canned crab meat", "canned shrimp", "canned oysters", "canned escargot", "canned beef", 
                              "canned chicken", "canned herring", "canned sprats", "frozen meat (chicken breasts, beef patties)", 
                              "frozen seafood (shrimp, fish fillets)", "frozen meatballs", "frozen sausage", "frozen hot dogs", 
                              "frozen burgers", "frozen chicken nuggets", "frozen fish sticks", "beef jerky", "turkey jerky"]

        # Convert user's ingredients to lowercase and trim whitespace
        ingredients_lower = [ingredient.lower().strip() for ingredient in ingredients]

        # Debug prints
        print("User's ingredients (lowercase):", ingredients_lower)
        print("Dietary choices (lowercase):", [dc.lower() for dc in dietary_choice])

        # Check for non-vegetarian ingredients and vegetarian dietary choice
        if "vegetarian" in [dc.lower() for dc in dietary_choice]:
            print("Vegetarian dietary choice detected.")
            if any(ingredient in non_vegetarian_ingredients for ingredient in ingredients_lower):
                print("Non-vegetarian ingredient detected.")
                return jsonify({'error': 'Non-vegetarian ingredients selected with vegetarian dietary choice. Please remove non-vegetarian ingredients.'}), 400

        prompt = generate_prompt(ingredients, num_people, dietary_choice, allergies)

        # Use asyncio to run the async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Generate the recipe asynchronously (First API Call)
        recipe_response = loop.run_until_complete(
            aclient.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "system", "content": "You are a professional and experienced chef."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
        )

        if not recipe_response.choices:
            return jsonify({'error': 'Failed to generate recipe'}), 500

        recipe_text = recipe_response.choices[0].message.content.strip()
        print("Generated recipe text:", recipe_text)  # Debug print

        if check_recipe_failure(recipe_text):
            print("Recipe generation failed.")  # Debug print
            return jsonify({'error': 'Recipe cannot be generated with provided combination of ingredients, allergens, and dietary restrictions'}), 400

        # Improved recipe name extraction with multiple patterns
        # name_patterns = [
        #     r"\*\*Recipe Name\*\*\n(.+?)\n\n",  # Pattern 1
        #     r"\*\*Recipe Name:\*\*\n(.+?)\n\n",
        #     r"^(.+?)\n\n",  # Pattern 2: Recipe name at the start
        #     r"^(.+?)\n",  # Pattern 3: Recipe name at the very beginning
        #     r"^\"(.+?)\"\n",  # Pattern 4: Recipe name within quotes
        # ]
        name_patterns = [
            r"\*\*Recipe Name\*\*\n(.+?)\n\n",  # Pattern 1
            r"^(.+?)\n\n",  # Pattern 2: Recipe name at the start
            r"^(.+?)\n",  # Pattern 3: Recipe name at the very beginning
            r"^\"(.+?)\"\n",  # Pattern 4: Recipe name within quotes
            r"^Recipe Name:\s*(.+)",  # Pattern 5: Recipe name with "Recipe Name:" prefix
        ]

        recipe_name = None
        for pattern in name_patterns:
            match = re.search(pattern, recipe_text)
            if match:
                recipe_name = match.group(1).strip()
                recipe_text = recipe_text.replace(match.group(0), "", 1).strip()
                break

        if not recipe_name:
            # Fallback: Use a default name if not found
            recipe_name = "Incorrect Recipe"

        # Remove enclosing ** from recipe name if present
        recipe_name = re.sub(r"^\*\*(.+?)\*\*$", r"\1", recipe_name)
        print("Extracted recipe name:", recipe_name)  # Debug print

        # Ensure the recipe name placeholder is replaced with the actual recipe name
        recipe_text = recipe_text.replace("Recipe Name", recipe_name)

        # Extract visualization prompt and remove it from the recipe text
        vis_prompt, recipe_text, vis_prompt_found = loop.run_until_complete(generate_image_prompt(recipe_text))
        print("Extracted visualization prompt:", vis_prompt)  # Debug print

        # Additional clean-up to ensure no visualization prompt remains
        recipe_text = re.sub(r"Visualization Prompt:.*", "", recipe_text).strip()
        recipe_text = re.sub(r"\*\*Visualization Prompt\*\*\n(.+)", "", recipe_text).strip()
        recipe_text = remove_nutritional_info(recipe_text)

        print("Recipe Text after remove_nutritional_info", recipe_text)


        if vis_prompt_found:
            # Add custom lines to the visualization prompt before calling fetch_image
            custom_prompt = f"Generate a realistic image of the prepared recipe according to the following: {vis_prompt}. Ensure the image is appetizing and well-presented."
            print("Custom visualization prompt:", custom_prompt)  # Debug print
            # Fetch the image asynchronously (Second API Call)
            image_result = loop.run_until_complete(fetch_image(custom_prompt))

            if 'error' in image_result:
                return jsonify({'error': image_result['error']}), 500

            # Download the generated image
            image_data = loop.run_until_complete(download_image(image_result['imageUrl']))

            # Upload image to Cloud Storage
            filename = f"images/{uuid.uuid4()}.png"
            cloud_storage_url = upload_to_cloud_storage(image_data, filename)

            # Create the response dictionary with the recipe name included
            response = {
                'name': recipe_name,
                'imageUrls': [cloud_storage_url],
                'recipe': recipe_text.strip()
            }
        else:
            # Use default prompt to generate image
            recipe_text = re.sub(r"Visualization Prompt:.*", "", recipe_text).strip()
            recipe_text = re.sub(r"\*\*Visualization Prompt\*\*\n(.+)", "", recipe_text).strip()
            recipe_text = remove_nutritional_info(recipe_text)
            default_image_prompt = f"Generate a realistic image of the prepared recipe according to the following: {recipe_text}. Ensure the image is appetizing and well-presented and should not contain any Nutritional information printed."
            print("Default visualization prompt:", default_image_prompt)  # Debug print
            image_result = loop.run_until_complete(fetch_image(default_image_prompt))

            if 'error' in image_result:
                return jsonify({'error': image_result['error']}), 500

            # Download the generated image
            image_data = loop.run_until_complete(download_image(image_result['imageUrl']))

            # Upload image to Cloud Storage
            filename = f"images/{uuid.uuid4()}.png"
            cloud_storage_url = upload_to_cloud_storage(image_data, filename)

            # Create the response dictionary with the recipe name included
            response = {
                'name': recipe_name,
                'imageUrls': [cloud_storage_url],
                'recipe': recipe_text.strip()
            }
        return jsonify(response)

    except openai.APIStatusError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
