import json
from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


def process_posts(raw_file_path, processed_file_path=None):
    try:
        with open(raw_file_path, encoding='utf-8') as file:
            posts = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading raw posts file: {e}")
        return

    enriched_posts = []
    for post in posts:
        # Check for 'caption' key instead of 'text'
        if 'caption' not in post:
            print(f"Skipping post as it does not contain the 'caption' key: {post}")
            continue

        try:
            metadata = extract_metadata(post['caption'])  # Use caption here
            post_with_metadata = post | metadata
            enriched_posts.append(post_with_metadata)
        except OutputParserException as e:
            print(f"Error extracting metadata for post: {post['caption'][:100]}... \nError: {e}")
            continue

    if not enriched_posts:
        print("No enriched posts available for processing.")
        return

    try:
        unified_tags = get_unified_tags(enriched_posts)
    except OutputParserException as e:
        print(f"Error unifying tags: {e}")
        return

    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags.get(tag, tag) for tag in current_tags}  # Fallback to original tag if not unified
        post['tags'] = list(new_tags)

    try:
        with open(processed_file_path, encoding='utf-8', mode="w") as outfile:
            json.dump(enriched_posts, outfile, indent=4)
    except IOError as e:
        print(f"Error writing processed posts file: {e}")


def clean_json_response(response):
    """Extract valid JSON from a response containing additional text."""
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        return json.loads(response[start:end])
    except json.JSONDecodeError:
        raise OutputParserException(f"Invalid JSON content:\n{response}")


def extract_metadata(post):
    template = '''
    You are given a Instagram post. You need to extract number of lines, language of the post, and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language, and tags. 
    3. tags is an array of text tags. Extract a maximum of two tags.
    4. Language should be English or Hinglish (Hinglish means Hindi + English)

    Here is the actual post on which you need to perform this task:  
    {post}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"post": post})

    try:
        return clean_json_response(response.content)
    except OutputParserException:
        print(f"Failed to parse response for post: {post[:100]}... \nResponse: {response.content}")
        raise


def get_unified_tags(posts_with_metadata):
    unique_tags = set()
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])

    if not unique_tags:
        raise OutputParserException("No tags found to unify.")

    unique_tags_list = ','.join(unique_tags)

    template = '''I will give you a list of tags. You need to unify tags with the following requirements:
    1. Tags are unified and merged to create a shorter list. 
       Example 1: "Self-Love & Confidence", "Love" can be merged into a single tag "Self-Love & Confidence". 
       Example 2: "Travel", "Adventure", "trip" can be mapped to "Travel&Adventure".
       Example 3: "Friendship", "Family", "Couples" can be mapped to "Family & Couples".
    2. Each tag should follow title case convention. Example: "Motivation", "Job Search".
    3. Output should be a JSON object. No preamble.
    4. Output should have a mapping of original tags to unified tags. 
       For example: {{"Love": "Self-Love & Confidence", "Travel": "Travel&Adventure", "Couple": "Family&Couples"}}

    Here is the list of tags: 
    {tags}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": str(unique_tags_list)})

    try:
        return clean_json_response(response.content)
    except OutputParserException:
        print(f"Failed to parse unified tags response. \nResponse: {response.content}")
        raise


if __name__ == "__main__":
    process_posts("data/raw_posts.json", "data/processed_posts.json")
