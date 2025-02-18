import os
import dotenv

OPENROUTERAI_BASE_URL = "https://openrouter.ai/api/v1"

if not os.path.exists('master.env'):
    raise Exception('master.env not found, please put your openrouter AI key inside: "OPENROUTERAI_API_KEY=sk-or-v1-a56691....."')
dotenv.load_dotenv('master.env')

OPENROUTERAI_API_KEY = os.getenv('OPENROUTERAI_API_KEY')

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
AURA_INSTANCEID = os.getenv('AURA_INSTANCEID')
AURA_INSTANCENAME = os.getenv('AURA_INSTANCENAME')