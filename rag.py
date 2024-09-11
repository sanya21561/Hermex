import time
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from together import Together

chatClient = Together(api_key='9277cfe863ae79d3063484d039ed2fa89681ecbfbe1477f3176b4f61638f04ef')
model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

client = QdrantClient(
    url="b57db28a-86e7-4be5-965e-67e98d7292eb.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="iD1_Lufho_ddBHK8d441fxRHyT8yc24yUeJLjHtvVFI4ABgJNd11qA",
)


embedding_model = TextEmbedding("snowflake/snowflake-arctic-embed-s")
data = pd.read_csv('dataset/flipkart_com-ecommerce_sample.csv')
collection_name = "flipGrid1"

class chatDetails:
    def __init__(self):
        self.chatHistory=[]
        self.productStatus=[]
        self.productHistory=[]
        self.cart=[]
    
    def clearCart(self):
        self.cart=[]
    
    def addToCart(self):
        self.cart.append(self.productHistory[-1])
        return self.productHistory[-1]
    
    def getProducStatus(self):
        return self.productStatus   
    
    def getCart(self):
        return self.cart
    
    def getChatHistory(self):
        return self.chatHistory
    
    def checkOut(self):
        for x in self.cart:
            self.productStatus.append([x, "Shipped", "reaching in 2 days", "Order ID: 123456"])
        self.cart=[]
        
    


def vectorSearch(query):
    query_embedding = np.squeeze(list(embedding_model.query_embed(query))).flatten()
    imf = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                ignore=False,
                rescore=False,
                oversampling=1,
            )
        )
    )
    # print(imf)
    return imf[2].id if imf else None

def runInferenceRetailSeller(model_name, question, embedding_model, client, collection_name, data, chatReport, flag):
    start = time.time()
    # flag=1 is Jerry , flag2 is Jessica
    # system_prompt = """You are a friendly, concise voice bot for an online marketplace. Your responses should be:
    # 1. Brief (30-50 words max)
    # 2. Conversational and engaging
    # 3. Focused on key product features and benefits
    # 4. Include a simple call to action or question to keep the conversation going
    # Avoid mentioning that you're an AI or voice bot."""
    prompt1="You are a Jerry, a customer service agent designed to assist customers with their inquiries, provide support, and resolve issues while remaining funny, yet professional. Your primary goals are to enhance customer satisfaction, ensure clear communication, and deliver accurate and helpful information. Do not produce false informations or hallucinations. If the customer asks to add something to cart, please add to cart. If customer wants to checkout say that you will checkout and ask if they need further assistance."
    prompt2="You are a Jessica, a customer service agent designed to assist customers with their inquiries, provide support, and resolve issues while remaining professional and empathetic. Your primary goals are to enhance customer satisfaction, ensure clear communication, and deliver accurate and helpful information. Do not produce false informations or hallucinations. Please ignore asterisks and replace Rs. as rupees. If the customer asks to add something to cart, please add to cart. If customer wants to checkout say that you will checkout and ask if they need further assistance."
    if flag==1:
        system_prompt=prompt1
    else:
        system_prompt=prompt2
    
    # Case 1: Customer asks for a product suggestion
    if "suggest" in question:
        product_id = vectorSearch(question)
        
        # print("product_id", product_id)
        if product_id is not None:
            row_i = data.iloc[product_id]
            product_info = {
                "name": row_i["product_name"],
                "price": row_i["discounted_price"],
                "description": row_i["description"],
                "brand": row_i["brand"],
                "url": row_i["product_url"]
            }
            chatReport.productHistory.append(product_info)
            user_prompt = f"""Customer asked: '{question}'
                Product: {product_info['name']}
                Price: Rs. {product_info['price']}
                Key features: {product_info['description'][:100]}

                Give a brief, friendly response about this product. End with a simple question or call to action."""

            response = chatClient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # max_tokens=75  # Limit the response length
            )
            latency = time.time() - start
            return response.choices[0].message.content.strip(), latency
        
    # Case 2: Customer asks for product details
    elif "cart"  in question and not "checkout" in question:
        if "add" in question:
            productAdded=chatReport.addToCart()
            user_prompt = f"""Customer asked: '{question}'
                Product: {productAdded['name']}
                Price: Rs. {productAdded['price']}
                Key features: {productAdded['description'][:100]}
                Tell the user that these are the products in his cart. """
        elif "clear" in question:
            chatReport.clearCart()
            user_prompt = f"""Customer asked: '{question}'
                Tell the user that their cart has been cleared"""
        elif "show" in question:
            cart = chatReport.getCart()
            if cart:
                user_prompt = f"""Customer asked: '{question}'
                    Your cart contains the following products:
                    {', '.join([p['name'] for p in cart])}
                    Would you like to add more products or proceed to checkout?"""
            else:
                user_prompt = f"""Customer asked: '{question}'
                    Your cart is empty. Would you like to add some products?"""
        else:
            user_prompt="I don't understand"
                
        response = chatClient.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # max_tokens=75  # Limit the response length
        )
        return response.choices[0].message.content.strip(), time.time() - start
        
    elif "update" in question:
        product_Status=chatReport.getProducStatus()
        if not product_Status:
            product_Status="No current orders"
        user_prompt = f"""Customer asked: '{question}'
            Your order status is as follows:
            {product_Status}
            Is there anything else I can help you with?"""
        response = chatClient.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # max_tokens=75  # Limit the response length
        )
        return response.choices[0].message.content.strip(), time.time() - start
        
    elif "checkout" or "check out" in question:
        chatReport.checkOut()
        user_prompt = f"""Customer asked: '{question}'
            Your order has been placed. Thank you for shopping with us!"""
        response = chatClient.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # max_tokens=75  # Limit the response length
        )
        
        return response.choices[0].message.content.strip(), time.time() - start
    else:
    #  chit chat
        response = chatClient.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            # max_tokens=75  # Limit the response length
        )
        return response.choices[0].message.content.strip(), time.time() - start
 