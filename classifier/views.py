# from django.shortcuts import render
# from django.http import JsonResponse
# from django.core.files.storage import FileSystemStorage
# from django.conf import settings
# import torch
# from transformers import ViTForImageClassification, ViTImageProcessor
# from PIL import Image
# import os
# import base64
# from io import BytesIO
# import requests
# from groq import Groq
# import json
# from django.shortcuts import render, redirect
# from django.contrib.auth.forms import UserCreationForm
# from django.contrib.auth import login
# # --------------------------------------------------
# # LOAD BOTH MODELS (STAGE + DISEASE)
# # --------------------------------------------------

# # Growth Stage Model
# STAGE_MODEL_PATH = os.path.join(settings.BASE_DIR, "vit_sunflower_model")
# stage_processor = ViTImageProcessor.from_pretrained(STAGE_MODEL_PATH)
# stage_model = ViTForImageClassification.from_pretrained(STAGE_MODEL_PATH)
# stage_model.eval()

# # Disease Detection Model
# DISEASE_MODEL_PATH = os.path.join(settings.BASE_DIR, "vit_disease_model")
# disease_processor = ViTImageProcessor.from_pretrained(DISEASE_MODEL_PATH)
# disease_model = ViTForImageClassification.from_pretrained(DISEASE_MODEL_PATH)
# disease_model.eval()

# device = "cuda" if torch.cuda.is_available() else "cpu"
# stage_model.to(device)
# disease_model.to(device)

# print(f"✅ Growth Stage Model loaded successfully on {device}")
# print(f"✅ Disease Detection Model loaded successfully on {device}")

# def index(request):
#     """Render the main page"""
# #     return render(request, 'classifier/index.html')
# def analysis(request):
#     """Render the analysis page with upload/capture interface"""
#     if request.method == 'GET':
#         return render(request, 'analysis_page.html')
#     elif request.method == 'POST':
#         # Handle POST requests through the predict function
#         return predict(request)

# def predict(request):
#     """Handle image prediction for both stage and disease"""
#     if request.method == 'POST':
#         try:
#             # Handle image input (camera or upload)
#             if 'camera_image' in request.POST:
#                 image_data = request.POST['camera_image']
#                 format, imgstr = image_data.split(';base64,')
#                 img_data = base64.b64decode(imgstr)
#                 image = Image.open(BytesIO(img_data)).convert('RGB')
#                 image_url = None
                
#             elif 'image' in request.FILES:
#                 image_file = request.FILES['image']
#                 fs = FileSystemStorage()
#                 filename = fs.save(image_file.name, image_file)
#                 image_url = fs.url(filename)
#                 image_path = os.path.join(settings.MEDIA_ROOT, filename)
#                 image = Image.open(image_path).convert('RGB')
#             else:
#                 return JsonResponse({'error': 'No image provided'}, status=400)
            
#             # --------------------------------------------------
#             # PREDICT GROWTH STAGE
#             # --------------------------------------------------
#             stage_inputs = stage_processor(images=image, return_tensors="pt")
#             stage_inputs = {k: v.to(device) for k, v in stage_inputs.items()}
            
#             with torch.no_grad():
#                 stage_outputs = stage_model(**stage_inputs)
            
#             stage_probs = torch.softmax(stage_outputs.logits, dim=1)
#             stage_confidence, stage_predicted = torch.max(stage_probs, dim=1)
            
#             stage_label = stage_model.config.id2label[stage_predicted.item()]
            
#             # Get all stage probabilities
#             all_stage_probs = {}
#             for idx, prob in enumerate(stage_probs[0].cpu().numpy()):
#                 class_name = stage_model.config.id2label[idx]
#                 all_stage_probs[class_name] = round(float(prob) * 100, 2)
            
#             # --------------------------------------------------
#             # PREDICT DISEASE
#             # --------------------------------------------------
#             disease_inputs = disease_processor(images=image, return_tensors="pt")
#             disease_inputs = {k: v.to(device) for k, v in disease_inputs.items()}
            
#             with torch.no_grad():
#                 disease_outputs = disease_model(**disease_inputs)
            
#             disease_probs = torch.softmax(disease_outputs.logits, dim=1)
#             disease_confidence, disease_predicted = torch.max(disease_probs, dim=1)
            
#             disease_label = disease_model.config.id2label[disease_predicted.item()]
            
#             # Get all disease probabilities
#             all_disease_probs = {}
#             for idx, prob in enumerate(disease_probs[0].cpu().numpy()):
#                 class_name = disease_model.config.id2label[idx]
#                 all_disease_probs[class_name] = round(float(prob) * 100, 2)
            
#             # --------------------------------------------------
#             # GET AI-POWERED TREATMENT PLAN
#             # --------------------------------------------------
#             treatment_plan = get_ai_treatment_plan(disease_label, stage_label)
            
#             # --------------------------------------------------
#             # RETURN BOTH PREDICTIONS WITH AI TREATMENT
#             # --------------------------------------------------
#             return JsonResponse({
#                 'success': True,
#                 'stage': {
#                     'predicted': stage_label,
#                     'confidence': round(float(stage_confidence.item()) * 100, 2),
#                     'all_probabilities': all_stage_probs
#                 },
#                 'disease': {
#                     'predicted': disease_label,
#                     'confidence': round(float(disease_confidence.item()) * 100, 2),
#                     'all_probabilities': all_disease_probs
#                 },
#                 'treatment': treatment_plan,
#                 'image_url': image_url
#             })
            
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
    
#     return JsonResponse({'error': 'Invalid request method'}, status=400)


# def get_weather(request):
#     """Get weather data by coordinates or city name"""
#     if request.method == 'GET':
#         try:
#             api_key = settings.OPENWEATHER_API_KEY
            
#             # Check if coordinates are provided (auto-detect)
#             lat = request.GET.get('lat')
#             lon = request.GET.get('lon')
#             city = request.GET.get('city')
            
#             if lat and lon:
#                 # Get weather by coordinates
#                 url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
#             elif city:
#                 # Get weather by city name
#                 url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
#             else:
#                 return JsonResponse({'error': 'Please provide location'}, status=400)
            
#             response = requests.get(url)
#             data = response.json()
            
#             if response.status_code == 200:
#                 weather_data = {
#                     'success': True,
#                     'temperature': round(data['main']['temp'], 1),
#                     'feels_like': round(data['main']['feels_like'], 1),
#                     'humidity': data['main']['humidity'],
#                     'description': data['weather'][0]['description'].title(),
#                     'icon': data['weather'][0]['icon'],
#                     'wind_speed': round(data['wind']['speed'] * 3.6, 1),  # Convert m/s to km/h
#                     'city': data['name'],
#                     'country': data['sys']['country']
#                 }
#                 return JsonResponse(weather_data)
#             else:
#                 return JsonResponse({'error': data.get('message', 'Unable to fetch weather')}, status=400)
                
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
    
#     return JsonResponse({'error': 'Invalid request method'}, status=400)

# def get_ai_treatment_plan(disease_name, growth_stage, weather_data=None):
#     """Get AI-generated treatment plan using Groq API"""
#     try:
#         client = Groq(api_key=settings.GROQ_API_KEY)
        
#         # Build context for AI
#         weather_context = ""
#         if weather_data:
#             weather_context = f"""
# Current Weather in {weather_data.get('city', 'the area')}:
# - Temperature: {weather_data.get('temperature', 'N/A')}°C
# - Humidity: {weather_data.get('humidity', 'N/A')}%
# - Conditions: {weather_data.get('description', 'N/A')}
# - Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h
# """
        
#         prompt = f"""You are an expert agricultural advisor specializing in sunflower cultivation in India.

# DETECTED INFORMATION:
# - Disease/Condition: {disease_name}
# - Growth Stage: {growth_stage}
# - Location: Bengaluru, Karnataka, India
# {weather_context}

# Please provide a comprehensive treatment plan in JSON format with the following structure:

# {{
#     "severity": "LOW/MEDIUM/HIGH/HEALTHY",
#     "immediate_actions": ["action 1", "action 2", "action 3"],
#     "disease_description": "Brief description of the condition",
#     "chemical_treatments": [
#         {{
#             "product_name": "Product name with Indian brands",
#             "dosage": "Exact dosage",
#             "frequency": "How often to apply",
#             "where_to_buy": "Agricultural stores in Bengaluru",
#             "approx_cost": "Cost in INR"
#         }}
#     ],
#     "organic_treatments": [
#         {{
#             "method": "Treatment method",
#             "recipe": "How to prepare",
#             "application": "How to apply"
#         }}
#     ],
#     "fertilizer_plan": {{
#         "primary": {{
#             "type": "NPK ratio and type",
#             "dosage": "Amount per plant/liter",
#             "frequency": "How often",
#             "purpose": "Why this fertilizer"
#         }},
#         "supplements": [
#             {{
#                 "type": "Supplement name",
#                 "dosage": "Amount",
#                 "purpose": "Benefit"
#             }}
#         ]
#     }},
#     "watering_advice": "Based on current weather, specific watering instructions",
#     "prevention_tips": ["tip 1", "tip 2", "tip 3"],
#     "recovery_timeline": "Expected recovery time and milestones",
#     "weather_specific_advice": "Advice based on current weather conditions",
#     "daily_care_schedule": {{
#         "morning": "What to do in morning",
#         "afternoon": "What to do in afternoon", 
#         "evening": "What to do in evening"
#     }}
# }}

# Focus on:
# 1. Products available in Bengaluru/Karnataka
# 2. Cost-effective solutions for small farmers
# 3. Both chemical and organic options
# 4. Weather-adjusted recommendations
# 5. Practical, actionable advice

# Return ONLY valid JSON, no additional text."""

#         # Call Groq API
#         chat_completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are an expert agricultural advisor. Always respond with valid JSON only."
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             model="llama-3.3-70b-versatile",
#             temperature=0.3,
#             max_tokens=2000
#         )
        
#         # Parse AI response
#         response_text = chat_completion.choices[0].message.content
        
#         # Try to extract JSON if wrapped in markdown code blocks
#         if "```json" in response_text:
#             response_text = response_text.split("```json")[1].split("```")[0].strip()
#         elif "```" in response_text:
#             response_text = response_text.split("```")[1].split("```")[0].strip()
            
#         treatment_plan = json.loads(response_text)
#         return treatment_plan
        
#     except Exception as e:
#         # Fallback to basic plan if AI fails
#         return {
#             "severity": "UNKNOWN",
#             "immediate_actions": ["Consult local agricultural expert"],
#             "disease_description": f"Unable to generate AI plan. Error: {str(e)}",
#             "chemical_treatments": [],
#             "organic_treatments": [],
#             "fertilizer_plan": {},
#             "watering_advice": "Water based on soil moisture",
#             "prevention_tips": ["Regular monitoring recommended"],
#             "recovery_timeline": "Consult expert for timeline",
#             "weather_specific_advice": "Monitor weather conditions",
#             "daily_care_schedule": {}
#         }
    
# def signup_view(request):
#     if request.method == 'POST':
#         form = UserCreationForm(request.POST)
#         if form.is_valid():
#             user = form.save()
#             login(request, user) # Automatically log the user in after signing up
#             return redirect('index') # Redirect to your home page
#     else:
#         form = UserCreationForm()
    
#     return render(request, 'accounts/signup.html', {'form': form})
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
import base64
from io import BytesIO
import requests
from groq import Groq
import json
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from datetime import datetime
import uuid
# --------------------------------------------------
# LOAD BOTH MODELS (STAGE + DISEASE)
# --------------------------------------------------

# Growth Stage Model
STAGE_MODEL_PATH = os.path.join(settings.BASE_DIR, "vit_sunflower_model")
stage_processor = ViTImageProcessor.from_pretrained(STAGE_MODEL_PATH)
stage_model = ViTForImageClassification.from_pretrained(STAGE_MODEL_PATH)
stage_model.eval()

# Disease Detection Model
DISEASE_MODEL_PATH = os.path.join(settings.BASE_DIR, "vit_disease_model")
disease_processor = ViTImageProcessor.from_pretrained(DISEASE_MODEL_PATH)
disease_model = ViTForImageClassification.from_pretrained(DISEASE_MODEL_PATH)
disease_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
stage_model.to(device)
disease_model.to(device)

print(f"✅ Growth Stage Model loaded successfully on {device}")
print(f"✅ Disease Detection Model loaded successfully on {device}")

def index(request):
    """Render the landing page"""
    return render(request, 'classifier/index.html')

def analysis(request):
    """Render the analysis page with upload/capture interface"""
    if request.method == 'GET':
        return render(request, 'classifier/analysis.html')
    elif request.method == 'POST':
        # Handle POST requests through the predict function
        return predict(request)

def predict(request):
    """Handle image prediction for both stage and disease"""
    if request.method == 'POST':
        try:
            # Handle image input (camera or upload)
            if 'camera_image' in request.POST:
                image_data = request.POST['camera_image']
                format, imgstr = image_data.split(';base64,')
                img_data = base64.b64decode(imgstr)
                image = Image.open(BytesIO(img_data)).convert('RGB')
                image_url = None
                
            elif 'image' in request.FILES:
                image_file = request.FILES['image']
                fs = FileSystemStorage()
                filename = fs.save(image_file.name, image_file)
                image_url = fs.url(filename)
                image_path = os.path.join(settings.MEDIA_ROOT, filename)
                image = Image.open(image_path).convert('RGB')
            else:
                return JsonResponse({'error': 'No image provided'}, status=400)
            
            # --------------------------------------------------
            # PREDICT GROWTH STAGE
            # --------------------------------------------------
            stage_inputs = stage_processor(images=image, return_tensors="pt")
            stage_inputs = {k: v.to(device) for k, v in stage_inputs.items()}
            
            with torch.no_grad():
                stage_outputs = stage_model(**stage_inputs)
            
            stage_probs = torch.softmax(stage_outputs.logits, dim=1)
            stage_confidence, stage_predicted = torch.max(stage_probs, dim=1)
            
            stage_label = stage_model.config.id2label[stage_predicted.item()]
            
            # Get all stage probabilities
            all_stage_probs = {}
            for idx, prob in enumerate(stage_probs[0].cpu().numpy()):
                class_name = stage_model.config.id2label[idx]
                all_stage_probs[class_name] = round(float(prob) * 100, 2)
            
            # --------------------------------------------------
            # PREDICT DISEASE
            # --------------------------------------------------
            disease_inputs = disease_processor(images=image, return_tensors="pt")
            disease_inputs = {k: v.to(device) for k, v in disease_inputs.items()}
            
            with torch.no_grad():
                disease_outputs = disease_model(**disease_inputs)
            
            disease_probs = torch.softmax(disease_outputs.logits, dim=1)
            disease_confidence, disease_predicted = torch.max(disease_probs, dim=1)
            
            disease_label = disease_model.config.id2label[disease_predicted.item()]
            
            # Get all disease probabilities
            all_disease_probs = {}
            for idx, prob in enumerate(disease_probs[0].cpu().numpy()):
                class_name = disease_model.config.id2label[idx]
                all_disease_probs[class_name] = round(float(prob) * 100, 2)
            
            # --------------------------------------------------
            # GENERATE REFERENCE ID AND STORE IN SESSION
            # --------------------------------------------------
            reference_id = f"AGRI-{datetime.now().strftime('%Y%m')}-{str(uuid.uuid4())[:8].upper()}"
            
            # Store data in session for treatment report page
            request.session['predicted_stage'] = stage_label
            request.session['predicted_disease'] = disease_label
            request.session['reference_id'] = reference_id
            request.session['stage_confidence'] = round(float(stage_confidence.item()) * 100, 2)
            request.session['disease_confidence'] = round(float(disease_confidence.item()) * 100, 2)
            request.session['image_url'] = image_url
            
            # --------------------------------------------------
            # RETURN BOTH PREDICTIONS (WITHOUT TREATMENT YET)
            # --------------------------------------------------
            return JsonResponse({
                'success': True,
                'stage': {
                    'predicted': stage_label,
                    'confidence': round(float(stage_confidence.item()) * 100, 2),
                    'all_probabilities': all_stage_probs
                },
                'disease': {
                    'predicted': disease_label,
                    'confidence': round(float(disease_confidence.item()) * 100, 2),
                    'all_probabilities': all_disease_probs
                },
                'image_url': image_url,
                'reference_id': reference_id
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)


def store_location(request):
    """Store location data in session"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Store in session
            request.session['location_data'] = {
                'latitude': data.get('latitude'),
                'longitude': data.get('longitude'),
                'city': data.get('city'),
                'country': data.get('country'),
                'temperature': data.get('temperature'),
                'humidity': data.get('humidity'),
                'description': data.get('description'),
                'wind_speed': data.get('wind_speed'),
                'feels_like': data.get('feels_like')
            }
            
            return JsonResponse({
                'success': True,
                'message': 'Location stored successfully'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    }, status=405)


def store_analysis(request):
    """Store analysis results in session"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Store in session
            request.session['analysis_results'] = {
                'disease': data.get('disease'),
                'stage': data.get('stage')
            }
            
            return JsonResponse({
                'success': True,
                'message': 'Analysis stored successfully'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    }, status=405)


def get_weather(request):
    """Get weather data by coordinates or city name"""
    if request.method == 'GET':
        try:
            api_key = settings.OPENWEATHER_API_KEY
            
            # Check if coordinates are provided (auto-detect)
            lat = request.GET.get('lat')
            lon = request.GET.get('lon')
            city = request.GET.get('city')
            
            if lat and lon:
                # Get weather by coordinates
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            elif city:
                # Get weather by city name
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            else:
                return JsonResponse({'error': 'Please provide location'}, status=400)
            
            response = requests.get(url)
            data = response.json()
            
            if response.status_code == 200:
                weather_data = {
                    'success': True,
                    'temperature': round(data['main']['temp'], 1),
                    'feels_like': round(data['main']['feels_like'], 1),
                    'humidity': data['main']['humidity'],
                    'description': data['weather'][0]['description'].title(),
                    'icon': data['weather'][0]['icon'],
                    'wind_speed': round(data['wind']['speed'] * 3.6, 1),  # Convert m/s to km/h
                    'city': data['name'],
                    'country': data['sys']['country'],
                    'latitude': data['coord']['lat'],
                    'longitude': data['coord']['lon']
                }
                return JsonResponse(weather_data)
            else:
                return JsonResponse({'error': data.get('message', 'Unable to fetch weather')}, status=400)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)


def find_nearby_stores(product_name, latitude, longitude, city_name, radius_km=10):
    """
    Find nearby agricultural stores selling the product using web search
    
    Args:
        product_name: Name of the agricultural product
        latitude: User's latitude
        longitude: User's longitude
        city_name: City name for search context
        radius_km: Search radius in kilometers (default 10km)
    
    Returns:
        List of nearby stores with details
    """
    try:
        # Search for stores selling this product near the location
        search_query = f"{product_name} agricultural store pesticide dealer {city_name}"
        
        # You can use Google Places API here for better results
        # For now, using a placeholder that you can replace with actual API call
        # Example with Google Places API:
        # places_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        # params = {
        #     'location': f"{latitude},{longitude}",
        #     'radius': radius_km * 1000,
        #     'keyword': f"agricultural store {product_name}",
        #     'key': settings.GOOGLE_PLACES_API_KEY
        # }
        # response = requests.get(places_url, params=params)
        # places_data = response.json()
        
        # Fallback: Return formatted search query for manual lookup
        # Replace this with actual Google Places API implementation
        return [
            {
                "store_name": f"Search: '{search_query}'",
                "address": f"Within {radius_km}km of your location",
                "distance": "Use Google Maps to find exact distance",
                "contact": "Search online for contact details"
            }
        ]
        
    except Exception as e:
        return [{
            "store_name": "Unable to find nearby stores",
            "address": f"Search '{product_name}' in {city_name}",
            "distance": "N/A",
            "contact": "N/A"
        }]


def get_ai_treatment_plan(disease_name, growth_stage, weather_data=None, location_data=None):
    """
    Get AI-generated treatment plan using Groq API with location-based store finder
    
    Args:
        disease_name: Detected disease or condition
        growth_stage: Current growth stage of the plant
        weather_data: Weather information dict
        location_data: Location information dict with 'latitude', 'longitude', 'city' (REQUIRED)
    
    Raises:
        ValueError: If location_data is not provided or missing required fields
    """
    try:
        # Validate location_data is provided
        if not location_data:
            raise ValueError("Location data is required for treatment plan generation")
        
        # Extract and validate location details
        latitude = location_data.get('latitude')
        longitude = location_data.get('longitude')
        city = location_data.get('city')
        
        if not all([latitude, longitude, city]):
            raise ValueError("Location data must include 'latitude', 'longitude', and 'city'")
        
        client = Groq(api_key=settings.GROQ_API_KEY)
        
        # Build context for AI
        weather_context = ""
        if weather_data:
            weather_context = f"""
Current Weather in {weather_data.get('city', city)}:
- Temperature: {weather_data.get('temperature', 'N/A')}°C
- Humidity: {weather_data.get('humidity', 'N/A')}%
- Conditions: {weather_data.get('description', 'N/A')}
- Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h
"""
        
        prompt = f"""You are an expert agricultural advisor specializing in sunflower cultivation in India.

DETECTED INFORMATION:
- Disease/Condition: {disease_name}
- Growth Stage: {growth_stage}
- Location: {city}, India
{weather_context}

Please provide a comprehensive treatment plan in JSON format with the following structure:

{{
    "severity": "LOW/MEDIUM/HIGH/HEALTHY",
    "immediate_actions": ["action 1", "action 2", "action 3"],
    "disease_description": "Brief description of the condition",
    "chemical_treatments": [
        {{
            "product_name": "Product name with Indian brands",
            "dosage": "Exact dosage",
            "frequency": "How often to apply",
            "approx_cost": "Cost in INR"
        }}
    ],
    "organic_treatments": [
        {{
            "method": "Treatment method",
            "recipe": "How to prepare",
            "application": "How to apply"
        }}
    ],
    "fertilizer_plan": {{
        "primary": {{
            "type": "NPK ratio and type",
            "dosage": "Amount per plant/liter",
            "frequency": "How often",
            "purpose": "Why this fertilizer"
        }},
        "supplements": [
            {{
                "type": "Supplement name",
                "dosage": "Amount",
                "purpose": "Benefit"
            }}
        ]
    }},
    "watering_advice": "Based on current weather, specific watering instructions",
    "prevention_tips": ["tip 1", "tip 2", "tip 3"],
    "recovery_timeline": "Expected recovery time and milestones",
    "weather_specific_advice": "Advice based on current weather conditions",
    "daily_care_schedule": {{
        "morning": "What to do in morning",
        "afternoon": "What to do in afternoon", 
        "evening": "What to do in evening"
    }}
}}

Focus on:
1. Products available in {city} area
2. Cost-effective solutions for small farmers
3. Both chemical and organic options
4. Weather-adjusted recommendations
5. Practical, actionable advice

Return ONLY valid JSON, no additional text."""

        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert agricultural advisor. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000
        )
        
        # Parse AI response
        response_text = chat_completion.choices[0].message.content
        
        # Try to extract JSON if wrapped in markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        treatment_plan = json.loads(response_text)
        
        # Add nearby store locations for each chemical treatment
        if "chemical_treatments" in treatment_plan:
            for treatment in treatment_plan["chemical_treatments"]:
                product_name = treatment.get("product_name", "")
                nearby_stores = find_nearby_stores(
                    product_name=product_name,
                    latitude=latitude,
                    longitude=longitude,
                    city_name=city
                )
                treatment["nearby_stores"] = nearby_stores
        
        # Add nearby store locations for fertilizers
        if "fertilizer_plan" in treatment_plan:
            if "primary" in treatment_plan["fertilizer_plan"]:
                fertilizer_type = treatment_plan["fertilizer_plan"]["primary"].get("type", "")
                nearby_stores = find_nearby_stores(
                    product_name=fertilizer_type,
                    latitude=latitude,
                    longitude=longitude,
                    city_name=city
                )
                treatment_plan["fertilizer_plan"]["primary"]["nearby_stores"] = nearby_stores
        
        return treatment_plan
        
    except ValueError as ve:
        # Re-raise validation errors
        raise ve
        
    except Exception as e:
        # Fallback to basic plan if AI fails
        return {
            "severity": "UNKNOWN",
            "immediate_actions": ["Consult local agricultural expert"],
            "disease_description": f"Unable to generate AI plan. Error: {str(e)}",
            "chemical_treatments": [],
            "organic_treatments": [],
            "fertilizer_plan": {},
            "watering_advice": "Water based on soil moisture",
            "prevention_tips": ["Regular monitoring recommended"],
            "recovery_timeline": "Consult expert for timeline",
            "weather_specific_advice": "Monitor weather conditions",
            "daily_care_schedule": {}
        }


def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user) # Automatically log the user in after signing up
            return redirect('index') # Redirect to your home page
    else:
        form = UserCreationForm()
    
    return render(request, 'accounts/signup.html', {'form': form})


def treatment_report(request):
    """Render the detailed AI treatment report page"""
    try:
        # Get location data from session
        location_data = request.session.get('location_data')
        
        if not location_data:
            return JsonResponse({
                'success': False,
                'error': 'Location data is required. Please set your location first.'
            }, status=400)
        
        # Get analysis results from session
        stage = request.session.get('predicted_stage')
        disease = request.session.get('predicted_disease')
        
        if not stage or not disease:
            return JsonResponse({
                'success': False,
                'error': 'Analysis results not found. Please analyze an image first.'
            }, status=400)
        
        # Prepare weather data
        weather_data = {
            'city': location_data.get('city'),
            'temperature': location_data.get('temperature'),
            'humidity': location_data.get('humidity'),
            'description': location_data.get('description'),
            'wind_speed': location_data.get('wind_speed'),
            'feels_like': location_data.get('feels_like')
        }
        
        # Prepare location info
        location_info = {
            'latitude': location_data.get('latitude'),
            'longitude': location_data.get('longitude'),
            'city': location_data.get('city'),
            'country': location_data.get('country')
        }
        
        # Generate AI treatment plan with location data
        treatment_data = get_ai_treatment_plan(
            disease_name=disease,
            growth_stage=stage,
            weather_data=weather_data,
            location_data=location_info
        )
        
        # Store treatment data in session
        request.session['treatment_data'] = treatment_data
        
        # Get other session data
        reference_id = request.session.get('reference_id', f"AGRI-{datetime.now().strftime('%Y%m')}-{str(uuid.uuid4())[:8].upper()}")
        stage_confidence = request.session.get('stage_confidence', 0)
        disease_confidence = request.session.get('disease_confidence', 0)
        
        # Generate current date
        generated_date = datetime.now().strftime('%B %d, %Y')
        
        context = {
            'stage': stage,
            'disease': disease,
            'treatment': treatment_data,
            'reference_id': reference_id,
            'generated_date': generated_date,
            'stage_confidence': stage_confidence,
            'disease_confidence': disease_confidence,
            'location': location_data,
            'weather': weather_data
        }
        
        return render(request, 'classifier/treatment_report.html', context)
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error generating treatment plan: {str(e)}'
        }, status=500)