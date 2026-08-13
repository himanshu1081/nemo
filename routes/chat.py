from fastapi import FastAPI, APIRouter
from langgraph.graph import StateGraph,START,END

router = APIRouter()

conversation_history=[]


@router.post("")
async def alexa(request: Request):

    print("Server hit on alexa skill")

    body = await request.json()

    request_type = body["request"]["type"]

    if request_type == "LaunchRequest":
        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": "Hello! I'm Nemo. What would you like to know?"
                },
                "shouldEndSession": False
            }
        }

    elif request_type == "IntentRequest":
        intent_name =body["request"]["intent"]["name"]

        if intent_name == "ChatIntent":
            query = body["request"]["intent"]["slots"]["message"]["value"]


            conversation_history.append(HumanMessage(content=query))

            systemPrompt = SystemMessage(content="You are Nemo AI running on Alexa. User has asked you question and you are supposed to answer them plus use appropriate tools possible. Keep replies short")

            reply =llm.invoke([systemPrompt]+conversation_history).content

            conversation_history.append(AIMessage(content=reply))

            return {
                "version": "1.0",
                "response": {
                    "outputSpeech": {
                        "type": "PlainText",
                        "text": reply
                    },
                    "shouldEndSession": False
                }
            }
