# src/api/load_tester.py
import asyncio
import random
import statistics
import time
from datetime import datetime

import aiohttp

API_URL = "http://127.0.0.1:8000/invocations" # 8000, to match the uvicorn server port

# The Chat Simulation Messages
PAYLOADS = [
    # NORMAL
    {"text": "This stream is amazing, great gameplay!", "type": "Normal"},
    {"text": "Salut, comment ça va chat?", "type": "Normal"},
    {"text": "Buen juego hermano", "type": "Normal"},
    {"text": "主播今天玩得真好", "type": "Normal"}, # Streamer played really well today
    
    # TOXIC
    {"text": "You are absolute trash, delete your account you garbage player.", "type": "Toxic"},
    {"text": "Tu es une merde absolue.", "type": "Toxic"},
    {"text": "Eres un pedazo de basura inútil.", "type": "Toxic"},
    {"text": "你是个彻头彻尾的白痴", "type": "Toxic"}, # You are an absolute idiot
    
    # BORDERLINE / PASSIVE AGGRESSIVE
    {"text": "Is this a joke? Because a toddler could play better than you.", "type": "Borderline"},
    {"text": "Are you actually crazy for doing this build?", "type": "Borderline"},
    {"text": "Est-ce une blague? Tu joues comme un enfant.", "type": "Borderline"},
    {"text": "你是不是脑子进水了？", "type": "Borderline"} # Is there water in your brain?
]

async def send_comment(session: aiohttp.ClientSession, comment_data: dict, req_id: int, latencies: list):
    """Fires a single asynchronous request to the FastAPI server and records latency."""
    start_time = time.perf_counter()
    
    try:
        async with session.post(API_URL, json={"text": comment_data["text"]}) as response:
            result = await response.json()
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
            
            # Confidence Interval Matching
            confidence = result.get("confidence", 0)
            if confidence >= 0.75:
                action = "🟥 AUTO-DELETE"
            elif confidence >= 0.50:
                action = "🟨 HUMAN REVIEW"
            else:
                action = "🟩 ALLOW"
                
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
                  f"Req_{req_id:03d} | Latency: {latency_ms:4.0f}ms | "
                  f"Type: {comment_data['type']:<10} | Action: {action:<15} | "
                  f"Conf: {confidence:.2f}")
            
    except Exception as e:  # noqa: BLE001
        print(f"Req_{req_id:03d} | ❌ FAILED | {e}")

async def twitch_chat_stream(comments_per_sec: int, duration_sec: int):
    """Simulates a continuous chat stream generating X concurrent requests every second."""
    print(f"\n Starting Chat Simulation: {comments_per_sec} comments/sec for {duration_sec} seconds...\n")
    
    latencies = []
    req_counter = 0
    
    async with aiohttp.ClientSession() as session:
        for second in range(duration_sec):
            print(f"--- Stream Second {second + 1} ---")
            
            # Generate the batch of comments for this specific second
            tasks = []
            for _ in range(comments_per_sec):
                req_counter += 1
                payload = random.choice(PAYLOADS)
                task = asyncio.create_task(send_comment(session, payload, req_counter, latencies))
                tasks.append(task)
            
            # Fire them concurrently
            await asyncio.gather(*tasks)
            
            # Wait until the next second begins to maintain the exact throughput rate
            await asyncio.sleep(1.0)
            
    # Calculating the final metrics
    print("\n" + "="*50)
    print("LOAD TEST ENGINEERING REPORT")
    print("="*50)
    print(f"Total Comments Processed : {len(latencies)}")
    if latencies:
        print(f"Mean Latency (Average)   : {statistics.mean(latencies):.1f} ms")
        print(f"Median Latency (P50)     : {statistics.median(latencies):.1f} ms")
        print(f"95th Percentile (P95)    : {statistics.quantiles(latencies, n=100)[94]:.1f} ms")
        print(f"Max Latency Spike        : {max(latencies):.1f} ms")
    print("="*50 + "\n")

if __name__ == "__main__":
    COMMENTS_PER_SECOND = 8
    TEST_DURATION_SECONDS = 5
    
    asyncio.run(twitch_chat_stream(COMMENTS_PER_SECOND, TEST_DURATION_SECONDS))