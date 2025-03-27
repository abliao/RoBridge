import os
import openai


def llm_plan(prompt):
    """
    
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = "gpt-4"
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    generated_text = response.choices[0].message.content
    # remove the first two lines from the generated text
    generated_text = generated_text.split('\n', 2)[2]
    # remove all numbers from the generated text
    generated_text = ''.join([i for i in generated_text if not i.isdigit()])
    # remove all periods from the generated text
    generated_text = generated_text.replace('.', '')
    # remove all spaces from the generated text
    generated_text = generated_text.replace(' ', '')
    # replace all commas with periods
    generated_text = generated_text.replace(',', '.')
    # remove all new lines from the generated text
    generated_text = generated_text.replace('\n', ', ')
    # split into a list of strings by a comma
    generated_text = generated_text.split(', ')
    # replace all periods with commas
    generated_text = [i.replace('.', ',') for i in generated_text]
    # convert each string element into a tuple
    generated_text = [eval(i) for i in generated_text]
    print(generated_text)
    return generated_text

if __name__ == "__main__":
    prompt = "Given the following operators: (grasp, place), objects: can, milk, bread, cereal. give me a plan to solve the following task: pick and place all objects. Make sure to take into account object geometry (tallest objects first)! can height: 10cm, milk height: 20 cm, cereal height: 30cm, bread height: 5cm. Formatting of output: a list in which each element looks like: (<object>, <operator>)"
    llm_plan(prompt)