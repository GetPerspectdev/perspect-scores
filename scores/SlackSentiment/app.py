import json
import slack_sdk
import pandas as pd
import gc
#from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import os
import requests
import sys
import tqdm
#from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode


## FROM PERSPECTIVE DOCUMENTATION

# allowed test types
allowed = ["TOXICITY",
           "SEVERE_TOXICITY",
           "TOXICITY_FAST",
           "ATTACK_ON_AUTHOR",
           "ATTACK_ON_COMMENTER",
           "INCOHERENT",
           "INFLAMMATORY",
           "OBSCENE",
           "OFF_TOPIC",
           "UNSUBSTANTIAL",
           "LIKELY_TO_REJECT",
           "INSULT"]

class Perspective(object):

    base_url = "https://commentanalyzer.googleapis.com/v1alpha1"

    def __init__(self, key):
        self.key = key

    def score(self, text, tests=["TOXICITY"], context=None, languages=None, do_not_store=False, token=None, text_type=None):
        # data validation
        # make sure it's a valid test
        # TODO: see if an endpoint that has valid types exists
        if isinstance(tests, str):
            tests = [tests]
        if not isinstance(tests, (list, dict)) or tests is None:
            raise ValueError("Invalid list/dictionary provided for tests")
        if isinstance(tests, list):
            new_data = {}
            for test in tests:
                new_data[test] = {}
            tests = new_data
        if text_type:
            if text_type.lower() == "html":
                text = remove_html(text)
            elif text_type.lower() == "md":
                text = remove_html(text, md=True)
            else:
                raise ValueError("{0} is not a valid text_type. Valid options are 'html' or 'md'".format(str(text_type)))

        for test in tests.keys():
            if test not in allowed:
                warnings.warn("{0} might not be accepted as a valid test.".format(str(test)))
            for key in tests[test].keys():
                if key not in ["scoreType", "scoreThreshhold"]:
                    raise ValueError("{0} is not a valid sub-property for {1}".format(key, test))

        # The API will only grade text less than 3k characters long
        if len(text) > 3000:
            # TODO: allow disassembly/reassembly of >3000char comments
            warnings.warn("Perspective only allows 3000 character strings. Only the first 3000 characters will be sent for processing")
            text = text[:3000]
        new_langs = []
        if languages:
            for language in languages:
                language = language.lower()
                if validate_language(language):
                    new_langs.append(language)

        # packaging data
        url = Perspective.base_url + "/comments:analyze"
        querystring = {"key": self.key}
        payload_data = {"comment": {"text": text}, "requestedAttributes": {}}
        for test in tests.keys():
            payload_data["requestedAttributes"][test] = tests[test]
        if new_langs != None:
            payload_data["languages"] = new_langs
        if do_not_store:
            payload_data["doNotStore"] = do_not_store
        payload = json.dumps(payload_data)
        headers = {'content-type': "application/json"}
        response = requests.post(url,
                            data=payload,
                            headers=headers,
                            params=querystring)
        data = response.json()
        if "error" in data.keys():
            raise PerspectiveAPIException(data["error"]["message"])
        c = Comment(text, [], token)
        base = data["attributeScores"]
        for test in tests.keys():
            score = base[test]["summaryScore"]["value"]
            score_type = base[test]["summaryScore"]["type"]
            a = Attribute(test, [], score, score_type)
            for span in base[test]["spanScores"]:
                beginning = span["begin"]
                end = span["end"]
                score = span["score"]["value"]
                score_type = span["score"]["type"]
                s = Span(beginning, end, score, score_type, c)
                a.spans.append(s)
            c.attributes.append(a)
        return c

class Comment(object):
    def __init__(self, text, attributes, token):
        self.text = text
        self.attributes = attributes
        self.token = token

    def __getitem__(self, key):
        if key.upper() not in allowed:
            raise ValueError("value {0} does not exist".format(key))
        for attr in self.attributes:
            if attr.name.lower() == key.lower():
                return attr
        raise ValueError("value {0} not found".format(key))

    def __str__(self):
        return self.text

    def __repr__(self):
        count = 0
        num = 0
        for attr in self.attributes:
            count += attr.score
            num += 1
        return "<({0}) {1}>".format(str(count/num), self.text)

    def __iter__(self):
        return iter(self.attributes)

    def __len__(self):
        return len(self.text)

class Attribute(object):
    def __init__(self, name, spans, score, score_type):
        self.name = name
        self.spans = spans
        self.score = score
        self.score_type = score_type

    def __getitem__(self, index):
        return self.spans[index]

    def __iter__(self):
        return iter(self.spans)

class Span(object):
    def __init__(self, begin, end, score, score_type, comment):
        self.begin = begin
        self.end = end
        self.score = score
        self.score_type = score_type
        self.comment = comment

    def __str__(self):
        return self.comment.text[self.begin:self.end]

    def __repr__(self):
        return "<({0}) {1}>".format(self.score, self.comment.text[self.begin:self.end])

class PerspectiveAPIException(Exception):
    pass

##### main function definition

def main():
    if len(sys.argv) != 2:
        print("Usage: python slack_toxicity_app.py YOUR_SLACK_API_KEY")
        return
    
    token_slack = sys.argv[1]

    ### get slack data
    # Create a client instance
    client = slack_sdk.WebClient(token=token_slack)

    # Get list of all direct message channels
    dm_channels_response = client.conversations_list(types=["im","mpim"])

    # Prepare a dictionary to store all messages
    all_messages = {}
    print('slack messages downloading...')
    # Iterate over each direct message channel
    for channel in tqdm.tqdm(dm_channels_response["channels"]):
        # Get conversation history
        history_response = client.conversations_history(channel=channel["id"])

        # Store messages
        all_messages[channel["id"]] = history_response["messages"]

    txts = []
    # progress_text = "Slack Messages Downloading. Please wait."
    # my_bar = st.progress(0, text=progress_text)
    print('slack messages processing...')
    for channel_id, messages in tqdm.tqdm(all_messages.items()):
        for message in messages:
            try:
                text = message["text"]
                user = message["user"]
                #toxicity_score = perspective.get_toxicity_score(text)
                timestamp = message["ts"]
                txts.append([timestamp,user,text])
                #toxicity_scores[timestamp] = toxicity_score
            except:
                pass

    df = pd.DataFrame(txts)
    df.columns =  ['timestamp','user','text']

    #### FILTER OUT TO ONLY INCLUDE USERS MESSAGES
    self_user = df['user'].value_counts().idxmax()
    df = df[df.user == self_user]




    comments = df['text']

    num_to_test = len(comments)

    #### run api
    google_api_key = 'AIzaSyCDlnuintUJAi1HKa4nAScA52T1gbn9v8g'
    client = Perspective(google_api_key)


    gc.collect()

    toxicity_scores = []
    insult_scores = []
    obscenity_scores = []

    print('finished pulling messages from slack. getting toxicity scores now.')

    for comment in tqdm.tqdm(enumerate(comments[:num_to_test])):

        try:
            toxicity = client.score(comment, tests=["TOXICITY", "INSULT", "OBSCENE"])
            
            toxicity_scores.append(toxicity["TOXICITY"].score)
            insult_scores.append(toxicity["INSULT"].score)
            obscenity_scores.append(toxicity["OBSCENE"].score)
        except:
            toxicity_scores.append(0)
            insult_scores.append(0)
            obscenity_scores.append(0)


    df_eda = df[0:num_to_test]
    df_eda['toxicity_scores'] = toxicity_scores
    df_eda['insult_scores'] = insult_scores
    df_eda['obscenity_scores'] = obscenity_scores
    df_eda['timestamp'] = pd.to_datetime(df_eda['timestamp'],unit='s')

    df_json = df_eda.to_json

    return df_json
    


if __name__ == "__main__":
    main()