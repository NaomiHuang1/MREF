import json
from openai import OpenAI

# 读取question choices 和 answer


with open("FineTuneMed.json", 'r') as file:
    data = json.load(file)
data = data[4835:]
api_key = "sk-gNpV2jX5PluMQVpTEc437cB3De6e46Dc89979338216495Ca"
base_url = "https://api.gpts.vin/v1"

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

chat_back = {}

out_file = 'gpt-llamaMedFinetune-4835-9999.json' 


for item in data:                        
    question_text = item["question"] 
    choice = item["choices"]
    reasoning = item['reasoning']
    answer = item['answer']
    v = "Question:"+ question_text+ "\n" + "Choices:" + choice + "\n"
    k = item['id']
    prompt_text = (
    "Below are some examples of MCQs with Reasoning-chain and their evaluations "
    "based on faithfulness, informativeness, repetition, coherence, and grammar. "
    "Criteria Definitions:\n"
    "- Faithfulness: Measures if the model misinterpreted the problem statement, or the reasoning chain is too vague, irrelevant, or misuses information. High scores require accurate alignment with the problem.\n"
    "- Informativeness: How well the reasoning provides detailed and useful information relevant to the question. High scores require thorough explanations.\n"
    "- Repetition: Measures repetition-related errors on the step level by checking if it paraphrases information already mentioned in the previous steps. No repetition warrants a perfect score.\n"
    "- Coherence: The logical flow and connectivity of ideas within the reasoning. High scores demand that all parts contribute effectively to the explanation.\n"
    "- Grammar: The grammatical correctness of the reasoning. Any grammatical errors should lead to a lower score.\n"
    "Each criterion should be rated from 0 to 1 with precision to four decimal places (e.g., 0.8765). \n"
    "Please evaluate strictly, looking for detail and precision in each aspect of the reasoning.\n\n"
    "### Example 1\n"
    "Question: What causes leaves to change color in the fall?\n"
    "Choices: A) Temperature changes B) Tree age C) Water levels D) Sun exposure\n"
    "Answer: A) Temperature changes\n"
    "Reasoning-chain: While the temperature is a significant factor in changing leaf colors, the explanation is missing specific details on how it affects different pigments like chlorophyll.\n"
    "Faithfulness: 0.7834\n"
    "Informativeness: 0.6589\n"
    "Repetition: 1.0000\n"
    "Coherence: 0.8417\n"
    "Grammar: 0.9783\n\n"
    "### Example 2\n"
    "Question: Why do we use solar panels?\n"
    "Choices: A) To generate electricity B) To heat water C) To power vehicles D) To reduce noise pollution\n"
    "Answer: A) To generate electricity\n"
    "Reasoning-chain: Solar panels are efficient in converting sunlight into electricity, but the response lacks details on the conversion efficiency and the types of panels.\n"
    "Faithfulness: 0.8119\n"
    "Informativeness: 0.4356\n"
    "Repetition: 0.9992\n"
    "Coherence: 0.8204\n"
    "Grammar: 0.9446\n\n"
    "### Please evaluate the Reasoning-chain for the following question based on the criteria defined above. (Do not need explanation) \n"
    f"Question: {question_text}\n"
    f"Choices: {choice}\n"
    f"Answer: {answer}\n"
    f"Reasoning-chains: {reasoning}\n"
    "\n"
    
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        model="gpt-4",
    )
    print(chat_completion.choices[0].message.content)
    chat_back[k] = v +  chat_completion.choices[0].message.content
    json_str = json.dumps(chat_back)
    with open(out_file, 'w') as json_file:
        json_file.write(json_str)


    