import re

from openai import OpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv(override=True)

def identify_assistant_speaker(dic_results: dict):
    """
    This function takes a dictionary of results from the speaker_transcription_and_identify function
    and returns the speaker identified as the assistant.


    Parameters:
    dic_results (dict): A dictionary containing the results

    Returns:
    str: The speaker identified as the assistant
    """
    # Create the client to ineract with OpenAI API
    client = OpenAI()

    ## Generate the system message to the AI
    with open('prompts/identify_assistant_speaker.prompt', 'r') as file:
        system_prompt = file.read()

    # Generate the user message
    user_message = f"This is the list of json's to analize:\n{str(dic_results)}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    ).choices[0].message.content

    ## Extract the speaker from the AI response
    # Regular expression to extract content inside curly braces
    pattern = r"\{(.*?)\}"

    # Extract the content inside the curly braces
    speaker = re.search(pattern, response).group(1)
    
    return speaker

# Example of Usage
if __name__ == "__main__":
    
    json = """[{'text': ' Hello?', 'speaker': 'SPEAKER_01'}, {'text': 'Hello.', 'speaker': 'SPEAKER_00'}, {'text': 'Yes, hello, sir.', 'speaker': 'SPEAKER_01'}, {'text': "May I speak with the business owner of Miguel's Restaurant?", 'speaker': 'SPEAKER_01'}, {'text': 'The owner.', 'speaker': 'SPEAKER_00'}, {'text': 'Pleasure speaking.', 'speaker': 'SPEAKER_01'}, {'text': 'Pleasure speaking to you, sir.', 'speaker': 'SPEAKER_01'}, {'text': "Apologies, I know you're busy.", 'speaker': 'SPEAKER_01'}, {'text': 'This is just less than a minute.', 'speaker': 'SPEAKER_01'}, {'text': 'Okay?', 'speaker': 'SPEAKER_01'}, {'text': 'So, my name here is Nats.', 'speaker': 'SPEAKER_01'}, {'text': "I'm from Aventus Bay.", 'speaker': 'SPEAKER_01'}, {'text': "And firstly, sir, I'm not here to change any of what you have.", 'speaker': 'SPEAKER_01'}, {'text': 'I respect what you have, okay?', 'speaker': 'SPEAKER_01'}, {'text': "This is regarding about the new guidelines because you're an owner.", 'speaker': 'SPEAKER_01'}, {'text': ' So basically, sir, owners are no longer required to pay any processing fees, which is good news.', 'speaker': 'SPEAKER_01'}, {'text': "It's zero processing.", 'speaker': 'SPEAKER_01'}, {'text': 'You will save thousands of dollars.', 'speaker': 'SPEAKER_01'}, {'text': 'But like I said, sir, this is just information that we want to share to you.', 'speaker': 'SPEAKER_01'}, {'text': "So my colleagues, sir, we're trying to definitely reach you back tomorrow for only a couple of minutes, and we will explain to you zero processing.", 'speaker': 'SPEAKER_01'}, {'text': 'So my question here, sir, is what would be the best time to reach you back tomorrow?', 'speaker': 'SPEAKER_01'}, {'text': 'Would it be in the morning, sir, or in the afternoon?', 'speaker': 'SPEAKER_01'}, {'text': "Yeah, I already added in the program that we don't pay processing fees.", 'speaker': 'SPEAKER_00'}, {'text': " Oh, that's good.", 'speaker': 'SPEAKER_01'}, {'text': 'What type of program do you use, sir?', 'speaker': 'SPEAKER_01'}, {'text': 'The one where the customers pay 3% or 3.25%.', 'speaker': 'SPEAKER_00'}, {'text': 'Oh, 3%.', 'speaker': 'SPEAKER_01'}, {'text': "There's still a percent, but this one, sir, this is zero processing.", 'speaker': 'SPEAKER_01'}, {'text': "There's no percentage.", 'speaker': 'SPEAKER_01'}, {'text': "But don't worry, sir.", 'speaker': 'SPEAKER_01'}, {'text': 'Like I said, I respect what you have.', 'speaker': 'SPEAKER_01'}, {'text': 'This is just information that we want to share to you.', 'speaker': 'SPEAKER_01'}, {'text': 'Who knows, maybe in the near future, if you will change, at least you will find us with zero, with no 3%, okay?', 'speaker': 'SPEAKER_01'}, {'text': 'So tomorrow, sir, would you like what time?', 'speaker': 'SPEAKER_01'}, {'text': 'In the morning or in the afternoon, sir?', 'speaker': 'SPEAKER_01'}, {'text': ' In the morning, you can call me in the morning.', 'speaker': 'SPEAKER_00'}, {'text': 'What time is good for you?', 'speaker': 'SPEAKER_00'}, {'text': 'Okay, 9 a.m.', 'speaker': 'SPEAKER_01'}, {'text': 'sir, 9 a.m.', 'speaker': 'SPEAKER_01'}, {'text': 'Is it okay?', 'speaker': 'SPEAKER_00'}, {'text': '9 a.m.', 'speaker': 'SPEAKER_00'}, {'text': "What's your name?", 'speaker': 'SPEAKER_00'}, {'text': 'I am Miguel.', 'speaker': 'SPEAKER_00'}, {'text': 'Call me at 9.30.', 'speaker': 'SPEAKER_00'}, {'text': '9.30, okay.', 'speaker': 'SPEAKER_00'}, {'text': 'M-I-G-U-L, yeah.', 'speaker': 'SPEAKER_00'}, {'text': 'Miguel, okay.', 'speaker': 'SPEAKER_01'}, {'text': "I know he's Hispanic.", 'speaker': 'SPEAKER_01'}, {'text': 'Yeah, my uncle is Hispanic.', 'speaker': 'SPEAKER_01'}, {'text': 'I know your spelling, sir, Miguel.', 'speaker': 'SPEAKER_01'}, {'text': 'So Miguel, we can call you back 9.30 tomorrow a.m.', 'speaker': 'SPEAKER_01'}, {'text': 'at 843-661-2990.', 'speaker': 'SPEAKER_01'}, {'text': 'Last question, Miguel.', 'speaker': 'SPEAKER_01'}, {'text': " I do believe with your success in the business, Miguel, you're processing at least 5,000 U.S.", 'speaker': 'SPEAKER_01'}, {'text': 'dollars a month.', 'speaker': 'SPEAKER_01'}, {'text': 'Just a guess.', 'speaker': 'SPEAKER_01'}, {'text': 'I believe so.', 'speaker': 'SPEAKER_01'}, {'text': 'Yes.', 'speaker': 'SPEAKER_01'}, {'text': 'Okay.', 'speaker': 'SPEAKER_01'}, {'text': 'Yes.', 'speaker': 'SPEAKER_00'}, {'text': 'Good.', 'speaker': 'SPEAKER_01'}, {'text': "So, Miguel, tomorrow, 9 p.m., don't forget, we'll call you.", 'speaker': 'SPEAKER_01'}, {'text': 'Okay?', 'speaker': 'SPEAKER_01'}, {'text': 'Have a great day, sir.', 'speaker': 'SPEAKER_01'}, {'text': 'God bless you.', 'speaker': 'SPEAKER_01'}, {'text': "You're so kind, sir.", 'speaker': 'SPEAKER_01'}, {'text': 'Thank you.', 'speaker': 'SPEAKER_01'}, {'text': 'Bye-bye.', 'speaker': 'SPEAKER_01'}]"""

    identify_assistant_speaker(json)
