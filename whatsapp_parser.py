from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import urllib
import emoji as emoji_util
import re
import spacy
from wordcloud import WordCloud, STOPWORDS
from nltk import ngrams
#import chart_studio.plotly as py
#import cufflinks as cf
#import plotly.express as px
#import plotly.graph_objects as go
from scipy.signal import find_peaks


# TODO:
    # assign self. variables from functions, redo docstrings
    # function for loading and parsing end to end
    # function to extract unique senders from chat

class Chat:
    def __init__(self, path_to_chat: str = ""):
        self.senders = []
        self.chat_raw = []
        self.chat_df = pd.DataFrame()
        self.chat_format = None
        self.path_to_chat = path_to_chat
        
    def load_chat_file(self, path_to_chat: str) -> List[str]:
        """
        Load a whatsapp chat export .txt file

        Parameters
        ----------
        path_to_chat : str
            Path to the .txt file of the whatsapp chat export.

        Returns
        -------
        List[str]
            List where every element is a single chat message.

        """
        
        with open(path_to_chat, 'r', encoding = 'UTF-8') as infile:
            # read whole file
            chat = infile.read()
            # split lines at newline (not an actual CRLF!)
            chat = chat.split('\n') #type(list)
            # remove \u200e lines since they are only contained in auto 
            # generated info messages from Whatsapp
            chat = [line for line in chat if r'\u200e' not in repr(line)]
            chat = [line for line in chat if "Sicherheitsnummer" not in line]
            # delete last entry because it contains only "\n"
            del chat[-1]
            
            self.chat_raw = chat #type(list)
        
    def determine_chat_format(self, chat_raw: list) -> str:
        """
        Find out what device was used to export the chat file.
        Android format is different from iOS in terms of timestamp.
        iOS: "[dd.mm.yy, HH:MM:SS]"
        Android: "dd.mm.yy, HH:MM:SS"
        Basicly the same but without the brackets.

        Parameters
        ----------
        chat_raw : list
            Takes chat_raw load via load_chat_file().

        Returns
        -------
        str
            returns "ios" or "android" depending on found format.

        """
    
        # take 20 random messages out of the chat and check their format
        ### ANDROID: 0; iOS: 1 ###
        
        result_list = []
        
        for n in np.random.randint(0, high=len(self.chat_raw), size=20):
            if str(self.chat_raw[n]).startswith('['):
                result_list.append(1)
            elif str(self.chat_raw[n])[0].isdigit():
                result_list.append(0)
            else:
                continue
        try:
            result = sum(result_list) / len(result_list)
        except ZeroDivisionError as e:
            raise UnknownChatFormat
    
        if result > 0.9:
            self.chat_format = "ios"
        elif result < 0.1:
            self.chat_format = "android"
        else:
            raise UnknownChatFormat()
        
    def check_message_integrity_ios(self.chat_raw: List[str]) -> List[str]:
        """
        Check if line is a valid ios message with timestamp, sender and message.
        Sometimes lines are cut of by CRLF respectively \n in this case.
        Put split messages back together in this case.

        Parameters
        ----------
        chat_raw : List[str]
            Takes chat_raw load via load_chat_file().

        Returns
        -------
        List[str]
            Returns integrity checked chat_raw.

        """
    
        ## if it's an iOS chat
        # check if all lines start with '[' and get indices of split messages
        split_messages_idx = [idx for idx, line in enumerate(chat_raw) \
                              if not line.startswith('[')]
    
        # make sure indices are sorted so chat list indices don't get messed
        # up when deleting indices
        for idx in sorted(split_messages_idx, reverse=True):
            # iterate over split messages and
            # merge them with the message send before
            merged_message = chat_raw[idx-1] + ' ' + chat_raw[idx] #type(str)
            chat_raw[idx-1] = merged_message
            
            # delete split messages by index after merging
            del chat_raw[idx]   
        
        return chat_raw #type(list)


    def check_message_integrity_android(chat_raw: List[str]) -> List[str]:
        """
        Check if line is a valid android message with timestamp, 
        sender and message.
        Sometimes lines are cut of by CRLF respectively \n in this case.
        Put split messages back together in this case.

        Parameters
        ----------
        chat_raw : List[str]
            Takes chat_raw load via load_chat_file().

        Returns
        -------
        List[str]
            Returns integrity checked chat_raw.

        """
    
        ## if it's an android chat
        # check if all lines start with a timestamp and
        # get indices of split messages
        split_messages_idx = []
        for idx, line in enumerate(chat_raw):
            try:
                datetime.datetime.strptime(line[:15], '%d.%m.%y, %H:%M')
            except ValueError:
                split_messages_idx.append(idx)
    
        # make sure indices are sorted so chat list indices don't get
        # messed up when deleting indices
        for idx in sorted(split_messages_idx, reverse=True):
            # iterate over split messages and merge them with the
            # message send before
            merged_message = chat_raw[idx-1] + ' ' + chat_raw[idx] #type(str)
            chat_raw[idx-1] = merged_message
            
            # delete split messages by index after merging
            del chat_raw[idx]
        
        return chat_raw #type(list)
    
    
    def parse_date_ios(line: str) -> datetime.datetime or str:
        """
        Takes single element of chat_raw of an ios chat_raw and returns date.

        Parameters
        ----------
        line : str
            single message (list element of chat_raw).

        Returns
        -------
        datetime.datetime
            returns datetime.datetime of chat message.

        """
        # split every line of chat between the first brackets
        try:
            date_string = line.split('[')[1].split(']')[0]
            # create datetime obj from remaining date format dd.mm.yy, HH:MM:SS
            message_date = datetime.datetime.strptime(date_string, '%d.%m.%y, %H:%M:%S') # type(datetime.datetime)
            return message_date
        except:
            return "unparsable"
        
    
    def parse_date_android(line: str) -> datetime.datetime or str:
        """
        Takes single element of chat_raw of an android chat_raw 
        and returns its date.

        Parameters
        ----------
        line : str
            single message (list element of chat_raw).

        Returns
        -------
        datetime.datetime
            returns datetime.datetime of chat message.

        """
        # split every line of chat after "-"
        date_string = line.split('-')[0].strip()
        # create datetime obj from remaining date format dd.mm.yy, HH:MM:SS
        try:
            message_date = datetime.datetime.strptime(date_string, '%d.%m.%y, %H:%M') # type(datetime.datetime)
            return message_date
        except IndexError:
            return "unparsable"
        
        
    def get_message_sender_android(line: str) -> str:
        """
        Get the sender of a single android chat message

        Parameters
        ----------
        line : str
            single message (list element of chat_raw).

        Returns
        -------
        str
            returns sender of the message.

        """
        # split string between "-" and ":" to get sender of the message
        try:
            return line.split('-')[1].split(':')[0].strip()
        except IndexError:
            return "unparsable"


    def get_message_sender_ios(line: str) -> str:
        """
        Get the sender of a single ios chat message

        Parameters
        ----------
        line : str
            single message (list element of chat_raw).

        Returns
        -------
        str
            returns sender of the message.

        """
        # split string between timestamp and ":" to get sender of the message
        try:
            return line.split(']')[1].split(':')[0].strip()
        except IndexError:
            return "unparsable"


    def chop_message_ios(line: str) -> str:
        """
        Extract the actual message from a raw ios message.
        Chop timestamp and sender to only get raw text.

        Parameters
        ----------
        line : str
            single message (list element of chat_raw).

        Returns
        -------
        str
            returns only the actual message.

        """
        # therefore split at 3rd ':', which indicates message start after sender tag
        try:
            return line.split(':', 3)[3].strip() #type(str)
        except:
            return "unparsable"

    def chop_message_android(line: str) -> str:
        """
        Extract the actual message from a raw android message.
        Chop timestamp and sender to only get raw text.

        Parameters
        ----------
        line : str
            single message (list element of chat_raw).

        Returns
        -------
        str
            returns only the actual message.

        """
        # therefore split at 2nd ':', which indicates message start after sender tag
        try:
            return line.split(':', 2)[2].strip() #type(str)
        except:
            return "unparsable"
        

    def get_emojis(message: str) -> List[any]:
        """
        Returns a list of all emojis in a message

        Parameters
        ----------
        message : str
            message element of chat_df.

        Returns
        -------
        List[any]
            returns list of all emojis used in the message.

        """
        emojis_in_message = [char for char in message \
                             if char in emoji_util.UNICODE_EMOJI['en']]
        return emojis_in_message

    
    def demojize_message(message: str) -> str:
        """
        Remove emojis from message text

        Parameters
        ----------
        message : str
            message element of chat_df.

        Returns
        -------
        str
            returns a demojized string.

        """
        ''''''
        # find all emojis in message
        emojis_in_message = [char for char in message if char in emoji_util.UNICODE_EMOJI['en']]
        # delete all given emojis from message
        for emoji in emojis_in_message:
            message = message.replace(emoji, '')
        
        return message


    def calc_time_diff(chat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate over chat_df and annotate time between messages

        Parameters
        ----------
        chat_df : pd.DataFrame
            The dataframe representation of the chat.

        Returns
        -------
        chat_df : pd.DataFrame
            The chat_df annotated with time between messages.

        """
        
        for idx, row in chat_df.iterrows():
    
            if idx == 0:
                # first message has no time diff
                continue
    
            timestamp_b = row['timestamp']
            timestamp_a = chat_df.iloc[idx-1]['timestamp']
            time_delta = timestamp_b - timestamp_a
            chat_df.at[idx, 'time_diff'] = time_delta
                    
        return chat_df


    def annotate_message_types(chat_df: pd.DataFrame,
                               chat_format: str,
                               answer_time_threshold: 
                                   datetime.timedelta=datetime.timedelta(days=2)
                               ) -> pd.DataFrame:
        
        """
        Iterate over chat_df and check if message is an answer, 
        follow up or new initiation.
        
        ################ EXAMPLE ####################
        ## [MSG_A] day1 10:00 sender_a: HI!
        ## [MSG B] day1 10:05 sender_b: Hi!
        ## --> then MSG B is an answer
        
        ## [MSG_A] day1 10:00 sender_a: bye!
        ## [MSG B] day9 12:05 sender_b: long time no see!
        ## --> then MSG B is a new initiation
        
        ## [MSG_A] day1 10:00 sender_a: hello?!?!?!
        ## [MSG B] day4 01:05 sender_a: hellooooooo?!
        ## --> then MSG B is also a new initiation
        
        ## [MSG_A] day1 10:00 sender_a: can you bring me something from the store?
        ## [MSG B] day1 12:05 sender_a: some milk and icecream!
        ## --> then MSG B is a follow up
        #############################################
        
        Parameters
        ----------
        chat_df : pd.DataFrame
            The dataframe representation of the chat.
            
        chat_format : str
            The format of the chat. Either "ios" or "android".
            
        answer_time_threshold : datetime.timedelta
            Threshold between two messages that differentiates a new initiation
            from an answer or follow up to a previous message.

        Returns
        -------
        chat_df : pd.DataFrame
            The chat_df annotated with message types.
        
        """
        
        if chat_format == 'ios':
            media_identification_msg = "Bild weggelassen"
        if chat_format == 'android':
            media_identification_msg = "<Medien ausgeschlossen>"
        
        for idx, row in chat_df.iterrows():
    
            if idx == 0:
                # first message is always an initiation and has no time diff or answer time
                chat_df.at[idx, 'message_type'] = "initiation"
                chat_df.at[idx, 'is_media'] = False
                continue
            
            # check if message is a picture or video
            if row['demojized_msg'] == media_identification_msg:
                chat_df.at[idx, 'is_media'] = True
                chat_df.at[idx, 'demojized_msg'] = ""
                chat_df.at[idx, 'message'] = ""
            else:
                chat_df.at[idx, 'is_media'] = False
            
            
            sender_b = row['sender']
            sender_a = chat_df.iloc[idx-1]['sender']
            time_delta = row['time_diff'] #type(datetime.timedelta)
    
            if sender_b == sender_a:
                if time_delta < answer_time_threshold:
                    # if sender_a and sender_b are the same and 
                    # time between the two messages is < answer_time_threshold,
                    # then it's a "follow up"
                    chat_df.at[idx, 'message_type'] = "follow_up"
    
                else:
                    # if between messages is > answer_time_threshold, 
                    # then it's a new initiation of the conversation
                    # the recipient didn't respond :(
                    chat_df.at[idx, 'message_type'] = "initiation"
    
    
            if sender_a != sender_b:
                if time_delta < answer_time_threshold:
                    # if sender_a and sender_b are NOT the same and
                    # time between the two messages < answer_time_threshold,
                    # then it's an answer
                    chat_df.at[idx, 'message_type'] = "answer"
                    chat_df.at[idx, 'answer_time_seconds'] = time_delta.seconds
    
                else:
                    # if time is > answer_time_threshold then it's a
                    # new initiation (or maybe just a sorry? ¯\_(ツ)_/¯)
                    chat_df.at[idx, 'message_type'] = "initiation"
    
        return chat_df


    def is_media(message: str) -> bool:
        """
        Returns True if the sent message was an image or video

        Parameters
        ----------
        message : str
            message element of chat_df.

        Returns
        -------
        bool
            Indicator if message was image or video.

        """

        if message == "Bild weggelassen" or message == "<Medien ausgeschlossen>":
            return True
        else:
            return False


    def get_ngrams(message: str, n: int) -> List[Tuple]:
        """
        Generate a list of tuples with all ngrams for a given input spacy doc

        Parameters
        ----------
        message : str
            Chat message in spacy doc format.
        n : int
            Number of n for n-grams.

        Returns
        -------
        List[Tuple]
            DESCRIPTION.

        """

        n_grams = ngrams(message, n)
        return [n for n in n_grams]
    

    def annotate_questions(spacy_doc: Any) -> bool:
        """
        Returns true if message is a question.

        Parameters
        ----------
        spacy_doc : Any
            Chat message in spacy doc format.

        Returns
        -------
        bool
            True if message is question.

        """
        if "?" in spacy_doc.text:
            return True
        else:
            return False


    def check_for_url(message: str) -> bool:
        """
        Returns True if the sent message contains URL

        Parameters
        ----------
        message : str
            message element of chat_df.

        Returns
        -------
        bool
            Returns True if the sent message contains URL.

        """
        
        if "www" in msg or "http" in msg:
            return True
        else:
            return False


    def get_url_domain(input_string: str) -> str:
        """
        Extract domains out of URLs in messages.

        Parameters
        ----------
        input_string : str
            message element of chat_df.

        Returns
        -------
        str
            Domain of the URL in the message.

        """
        domain = urllib.parse.urlparse(input_string).netloc
        # sometimes urlparse() returns empty string, try to parse by simple split then
        if domain == '':
            try:
                return input_string.split('https://')[1].split('/')[0]
            except IndexError:
                try:
                    return input_string.split('http://')[1].split('/')[0]
                except IndexError:
                    return None
        
        else:
            return domain


    def get_lemma(token: spacy.tokens.token.Token) -> str:
        """
        Takes spacy token and return lemma of the token, if possible.
        If not, returns string representation of the token.

        Parameters
        ----------
        token : spacy.tokens.token.Token
            DESCRIPTION.

        Returns
        -------
        str
            Returns lemma if exists. Else returns just the text.

        """
        
        if token.lemma_:
            return token.lemma_
        else:
            return token.text

    @staticmethod
    def parse_chat(path_to_chat: str) -> pd.DataFrame:
        start_time = datetime.datetime.now()
        chat = load_chat(path_to_chat) #type(list)
        chat_format = determine_chat_format(chat) #type(str)
        
        if chat_format.lower() == "android":
            chat = check_message_integrity_android(chat) #type(list)
        elif chat_format.lower() == "ios":
            chat = check_message_integrity_ios(chat) #type(list)
        
        chat_df = pd.DataFrame(chat, columns=['message_raw'])
        print(f"df time: {datetime.datetime.now() - start_time}")
    
        cache_time = datetime.datetime.now()
        if chat_format.lower() == 'ios':
            chat_df['sender'] = chat_df['message_raw']\
                .apply(lambda x: get_message_sender_ios(x))
                
            chat_df['timestamp'] = chat_df['message_raw']\
                .apply(lambda x: parse_date_ios(x))
                
            chat_df['message'] = chat_df['message_raw']\
                .apply(lambda x: chop_message_ios(x))
    
        else:
            chat_df['sender'] = chat_df['message_raw']\
                .apply(lambda x: get_message_sender_android(x))
                
            chat_df['timestamp'] = chat_df['message_raw']\
                .apply(lambda x: parse_date_android(x))
                
            chat_df['message'] = chat_df['message_raw']\
                .apply(lambda x: chop_message_android(x))
    
        # drop messages without valid timestamp (parsing errors) 
        chat_df = chat_df[chat_df['timestamp']\
                          .apply(lambda x: isinstance(x, datetime.datetime))]
        chat_df = chat_df.sort_values('timestamp').reset_index(drop=True)
        
        # further timestamp extractions
        cache_time = datetime.datetime.now()
        chat_df['date'] = chat_df['timestamp'].apply(lambda x: x.date())
        chat_df['time'] = chat_df['timestamp'].apply(lambda x: x.time())
        chat_df['weekday'] = chat_df['timestamp']\
            .apply(lambda x: x.strftime('%A'))
        chat_df['hour'] = chat_df['time'].apply(lambda x: x.hour)
        print(
            f"timestamp annotations time: "\
            f"{datetime.datetime.now() - cache_time}")
        
        # extract emoji stuff
        cache_time = datetime.datetime.now()
        chat_df['emojis'] = chat_df['message'].apply(lambda x: get_emojis(x))
        print(f"emoji extract time: {datetime.datetime.now() - cache_time}")
        cache_time = datetime.datetime.now() 
        chat_df['demojized_msg'] = chat_df['message']\
            .apply(lambda x: demojize_message(x))
        print(f"demojize time: {datetime.datetime.now() - cache_time}")
    
        # annotations for message type and answer time
        cache_time = datetime.datetime.now()
        chat_df['time_diff'] = chat_df['timestamp'] - chat_df['timestamp'].shift(periods=1)
        chat_df = annotate_message_types(chat_df, chat_format)
        chat_df['is_media'] = chat_df['message'].apply(lambda x: is_media(x))
        print(f"message annotations time: {datetime.datetime.now() - cache_time}")
    
        # parse urls and get domains
        cache_time = datetime.datetime.now()
        chat_df['has_url'] = chat_df['message'].apply(check_for_url)
        chat_df['url_domain'] = chat_df[chat_df['has_url'] == True]\
            ['message'].apply(lambda x: get_url_domain(x))
        print(f"url parse and annotations time: "\
              f"{datetime.datetime.now() - cache_time}")
    
        # initialize NLP model
        cache_time = datetime.datetime.now()
        nlp = spacy.load("de_core_news_md",
                         exclude=['senter', 'sentencizer', 'attribute_ruler',
                                  'parser', 'morphologizer', 'ner'])
        nlp.remove_pipe("ner")
        nlp.remove_pipe("parser")
    
        # add stopwords by hand
        with open('stopwords_ger.txt', 'r', encoding='UTF-8') as infile:
            stopwords = infile.read().splitlines()
    
        for w in stopwords:
            nlp.vocab[w].is_stop = True
    
        # NLP parsing
        chat_df['spacy_doc'] = chat_df['demojized_msg']\
            .apply(lambda x: nlp(x.lower()))
            
        chat_df['words'] = chat_df['spacy_doc']\
            .apply(lambda x: [token.text for token in x])
        chat_df['nouns'] = chat_df['spacy_doc']\
            .apply(lambda doc: [token.text for token in doc
                                if token.pos_ == "NOUN" 
                                and token.is_stop == False])
        chat_df['verbs'] = chat_df['spacy_doc']\
            .apply(lambda doc: [token.text for token in doc 
                                if token.pos_ == "VERB" 
                                and token.is_stop == False])
            
        chat_df['lemmas'] = chat_df['spacy_doc']\
            .apply(lambda doc: [token.lemma_ for token in doc 
                                if token.is_stop == False 
                                and token.is_punct == False
                                and token.lemma_ != " "])
        
        chat_df['is_question'] = chat_df['spacy_doc'].apply(annotate_questions)
    
        # generate ngrams
        chat_df['trigrams'] = chat_df['lemmas'].apply(get_ngrams, n=3)
        chat_df['bigrams'] = chat_df['lemmas'].apply(get_ngrams, n=2)
        print(f"NLP time: {datetime.datetime.now() - cache_time}")
    
        print(f"took {datetime.datetime.now() - start_time}")
        return chat_df




class UnknownChatFormat(Exception):
    """
    Raised when chat format could not be detected
    """
    
    def __init__(self, message="Unknown chat format: could not detect if \
                 android or iOS was used"):
        self.message = message
        super().__init__(self.message)
        
        
        
chat_ = Chat()
chat_.load_chat_file('Schimpfwortliste.txt')
