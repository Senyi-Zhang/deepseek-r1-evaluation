from openai import OpenAI


def deepseek_predict(system_message='', user_message='', api_key =''):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        stream=False
    )

    return response.choices[0].message.content

