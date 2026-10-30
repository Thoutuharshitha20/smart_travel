import os
from html import escape
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

APP_TITLE = "Smart Travel Planning Agent"
DEFAULT_MODEL = os.getenv("MODEL_ID", "gemma-4-31b-it")
ALLOWED_MODELS = {"gemma-4-31b-it", "gemini-3.8-flash"}

app = FastAPI(title=APP_TITLE, version="1.0.0")


class TravelRequest(BaseModel):
    region: str = Field(..., min_length=2, max_length=100)
    budget: float = Field(..., gt=0, le=100000000)
    days: int = Field(..., ge=1, le=30)
    interests: str = Field(..., min_length=2, max_length=500)
    travelers: int = Field(default=1, ge=1, le=50)
    model: Optional[str] = None


def get_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Set it as an environment variable."
        )
    return genai.Client(api_key=api_key)


def generate_text(prompt: str, model_name: str) -> str:
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            "Unsupported model. Choose gemma-4-31b-it or gemini-3.8-flash."
        )

    client = get_client()
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=4000,
        ),
    )

    result = getattr(response, "text", None)
    if not result:
        raise RuntimeError("The model returned an empty response.")
    return result.strip()


def destination_research_agent(data: TravelRequest, model_name: str) -> str:
    prompt = f"""
You are Agent 1, the Destination Research Agent in a Smart Travel Planning
system. Recommend destinations and attractions based on the user's needs.

USER DETAILS
Region or preferred area: {data.region}
Total trip budget in INR: Rs. {data.budget:,.0f}
Trip duration: {data.days} days
Number of travellers: {data.travelers}
Interests: {data.interests}

TASKS
1. Suggest up to 3 suitable destinations in or relevant to the requested region.
2. Explain briefly why each destination matches the interests.
3. List suitable attractions and activities.
4. Give approximate cost categories for stay, food, local transport and entry fees.
5. Recommend one best-fit destination.
6. Keep the total budget and trip duration in mind.
7. If the budget seems too low, say so honestly and suggest a simpler option.

IMPORTANT
- Use Indian Rupees (INR).
- Do not claim that you checked live hotel, flight, train or ticket prices.
- Do not invent current opening hours, availability or exact fares.
- Mark all costs as rough estimates that the traveller must verify.
- Make the output clear and easy to read.
"""
    return generate_text(prompt, model_name)


def travel_planning_agent(
    data: TravelRequest,
    destination_research: str,
    model_name: str,
) -> str:
    prompt = f"""
You are Agent 2, the Smart Travel Planning Agent. You receive the research
from Agent 1 and turn it into a practical trip plan.

USER DETAILS
Region: {data.region}
Total budget: Rs. {data.budget:,.0f}
Duration: {data.days} days
Travellers: {data.travelers}
Interests: {data.interests}

AGENT 1 DESTINATION RESEARCH
{destination_research}

YOUR TASKS
1. Choose the most suitable destination from Agent 1's research.
2. Create a day-wise itinerary for exactly {data.days} days.
3. For each day, give morning, afternoon and evening activities.
4. Include practical travel tips and sensible activity grouping.
5. Estimate accommodation, food, local transport and sightseeing costs
   for the whole group of {data.travelers} traveller(s).
6. Add the categories to calculate a total estimated trip cost.
7. Compare the estimated total with the user's budget of Rs. {data.budget:,.0f}.
8. Clearly state "Within budget" or "Over budget".
9. If over budget, suggest specific ways to reduce costs.
10. Mention any assumptions, such as budget excluding travel to the destination.

OUTPUT HEADINGS
- Selected Destination
- Why This Destination
- Day-wise Itinerary
- Estimated Expense Breakdown
- Total Estimated Cost
- Budget Status
- Money-saving Suggestions
- Travel Tips

IMPORTANT
- Use INR and readable formatting.
- All prices are estimates, not live or guaranteed prices.
- Do not claim to book anything or verify live availability.
- Do not invent exact opening hours or transport schedules.
"""
    return generate_text(prompt, model_name)


def run_two_agents(data: TravelRequest) -> dict:
    model_name = data.model or DEFAULT_MODEL
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            "Select either gemma-4-31b-it or gemini-3.8-flash."
        )

    # Agent 1 researches destinations.
    research = destination_research_agent(data, model_name)

    # Agent 2 uses Agent 1's result to build the itinerary.
    itinerary = travel_planning_agent(data, research, model_name)

    return {
        "model": model_name,
        "agent_1_research": research,
        "agent_2_travel_plan": itinerary,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Travel Planning Agent</title>
<style>
body{font-family:Arial,sans-serif;background:#f3f6fb;margin:0;padding:24px;color:#172033}
.wrap{max-width:900px;margin:auto}
.card{background:white;padding:24px;border-radius:14px;box-shadow:0 4px 18px #00000012;margin-bottom:18px}
h1{color:#174ea6;margin-top:0}
label{display:block;font-weight:600;margin:13px 0 6px}
input,textarea,select{width:100%;box-sizing:border-box;padding:11px;border:1px solid #ccd4e0;border-radius:8px;font-size:15px}
textarea{min-height:85px}
button{background:#1759b5;color:white;border:0;padding:12px 20px;border-radius:8px;font-size:16px;cursor:pointer;margin-top:16px}
button:disabled{opacity:.6;cursor:wait}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:Arial,sans-serif;line-height:1.55}
.result{border-top:1px solid #e3e8ef;padding-top:16px;margin-top:16px}
.small{color:#5e6878;font-size:13px}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Smart Travel Planning Agent</h1>
    <p>Two AI agents work together to research destinations and prepare your travel itinerary.</p>
    <form id="travelForm">
      <label>Region / preferred area</label>
      <input id="region" value="South India" required>
      <label>Total budget (INR)</label>
      <input id="budget" type="number" min="1" value="15000" required>
      <label>Number of days</label>
      <input id="days" type="number" min="1" max="30" value="3" required>
      <label>Number of travellers</label>
      <input id="travelers" type="number" min="1" value="1" required>
      <label>Interests</label>
      <textarea id="interests" required>Beaches, historical places and local food</textarea>
      <label>AI model</label>
      <select id="model">
        <option value="gemma-4-31b-it">Gemma 4 31B IT</option>
        <option value="gemini-3.8-flash">Gemini 3.8 Flash</option>
      </select>
      <button id="submitBtn" type="submit">Generate My Travel Plan</button>
    </form>
    <p class="small">Costs are AI-generated estimates, not live prices or bookings.</p>
    <div id="status"></div>
  </div>
  <div class="card" id="results" style="display:none">
    <h2>Agent 1 — Destination Research</h2>
    <pre id="research"></pre>
    <div class="result">
      <h2>Agent 2 — Final Travel Plan</h2>
      <pre id="plan"></pre>
    </div>
  </div>
</div>
<script>
const form = document.getElementById("travelForm");
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const btn = document.getElementById("submitBtn");
  const status = document.getElementById("status");
  const results = document.getElementById("results");
  btn.disabled = true;
  status.textContent = "Agent 1 is researching destinations, then Agent 2 will prepare your itinerary...";
  results.style.display = "none";

  const payload = {
    region: document.getElementById("region").value,
    budget: Number(document.getElementById("budget").value),
    days: Number(document.getElementById("days").value),
    travelers: Number(document.getElementById("travelers").value),
    interests: document.getElementById("interests").value,
    model: document.getElementById("model").value
  };

  try {
    const response = await fetch("/plan", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Request failed");
    document.getElementById("research").textContent = data.agent_1_research;
    document.getElementById("plan").textContent = data.agent_2_travel_plan;
    results.style.display = "block";
    status.textContent = "Travel plan generated successfully.";
  } catch (error) {
    status.textContent = "Error: " + error.message;
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
""")


@app.get("/health")
def health():
    return {"status": "ok", "application": APP_TITLE}


@app.post("/plan")
def create_plan(data: TravelRequest):
    try:
        return run_two_agents(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # Keep detailed secrets/API responses out of the browser.
        print("Travel planning error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Travel plan generation failed. Check the API key, model access and server logs."
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
