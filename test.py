import google.generativeai as genai

genai.configure(api_key="AIzaSyDa1OfiVFfBB9LOmLRVlL5tjR-hSkVHD3A")

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content("Xin chào, bạn khỏe không?")
print(response.text)
