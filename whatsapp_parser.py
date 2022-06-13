from typing import List, Tuple, Any
import pandas as pd
import numpy as np
import datetime
import urllib
import emoji as emoji_util
import spacy
import re
from nltk import ngrams


class Chat:
    def __init__(self, path_to_chat: str, nlp_parsing: bool = False, chat_language: str = 'german'):
        self.senders = []
        self.chat_raw = []
        self.chat_df = pd.DataFrame()
        self.chat_format = ""
        self.path_to_chat = path_to_chat
        self.nlp = nlp_parsing
        self.language = chat_language

        if self.language in ('deutsch', 'german', 'deu', 'ger'):
            self.lang_dict = {'media_identification_messages_ios': [
                'bild weggelassen',
                'video weggelassen',
                'gif weggelassen',
                'audio weggelassen'
            ],
                'nlp_model_name': 'de_core_news_md',
                'stopword_file_name': 'stopwords_ger.txt'}

        elif self.language.lower() in ('englisch', 'english', 'eng', 'us'):
            self.lang_dict = {'media_identification_messages_ios': [
                'image omitted',
                'video omitted',
                'gif omitted',
                'audio omitted'
            ],
                'image_identification_message_android': '.jpg (',
                'nlp_model_name': 'en_core_web_md',
                'stopword_file_name': 'stopwords_eng.txt'}
        else:
            raise ValueError('No proper language give. Only english and german supported')

    def load_chat_file(self):
        """
        Load a whatsapp chat export .txt file
        """

        with open(self.path_to_chat, 'r', encoding='UTF-8') as infile:
            # read whole file
            chat = infile.read()
            # split lines at newline (not an actual CRLF!)
            chat = chat.split('\n')  # List[str]
            # remove invisible unicode formatting characters
            # there might be more in some chats
            for idx, line in enumerate(chat.copy()):
                if '\u200e' in line:
                    chat[idx] = line.replace('\u200e', '')
                if '\u200d' in line:
                    chat[idx] = line.replace('\u200d', '')
            # delete last entry because it contains only "\n"
            del chat[-1]

        self.chat_raw = chat  # List[str]

    def determine_chat_format(self):
        """
        Find out what device was used to export the chat file.
        Android format is different from iOS in terms of timestamp.
        iOS: "[dd.mm.yy, HH:MM:SS]"
        Android: "dd.mm.yy, HH:MM:SS"
        """

        # take 20 random messages out of the chat and check their format
        # ANDROID: 0; iOS: 1
        if len(self.chat_raw) == 0:
            raise Warning("chat has not been loaded yet or is empty or unreadable")

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
        except ZeroDivisionError:
            raise UnknownChatFormat

        if result > 0.9:
            self.chat_format = "ios"
        elif result < 0.1:
            self.chat_format = "android"
        else:
            raise UnknownChatFormat

    def check_message_integrity_ios(self):
        """
        Check if line is a valid ios message with timestamp, sender and message.
        Sometimes lines are split by CRLF, respectively \n in this case.
        Put split messages back together in this case.
        """

        # get indices of split messages which do not start with '['
        split_messages_idx = [idx for idx, line in enumerate(self.chat_raw)
                              if not line.startswith('[')]

        # iterate from bottom to top to not mess up indices
        # when deleting entries
        for idx in sorted(split_messages_idx, reverse=True):
            # iterate over split messages and
            # merge them with the message send before
            merged_message = self.chat_raw[idx - 1] + ' ' + self.chat_raw[idx]  # str
            self.chat_raw[idx - 1] = merged_message

            # delete original split message by index after merging
            del self.chat_raw[idx]

        # dealing with Whatsapp's own system messages is tricky because they mess up statistics and stuff
        # they don't have a sender so can easily detect them because they only have two ":" without the "sender"-part
        self.chat_raw = [msg for msg in self.chat_raw if msg.count(':') > 2]

    def check_message_integrity_android(self):
        """
        Check if line is a valid android message with timestamp, 
        sender and message.
        Sometimes lines are cut off by CRLF, respectively \n in this case.
        Put split messages back together in this case.
        """

        # check if all lines start with a timestamp and
        # get indices of split messages
        split_messages_idx = []
        for idx, line in enumerate(self.chat_raw):
            try:
                datetime.datetime.strptime(line[:15], '%d.%m.%y, %H:%M')
            except ValueError:
                split_messages_idx.append(idx)

        # iterate from bottom to top to not mess up indices
        # when deleting entries
        for idx in sorted(split_messages_idx, reverse=True):
            # iterate over split messages and merge them with the
            # message send before
            merged_message = self.chat_raw[idx - 1] + ' ' + self.chat_raw[idx]  # str
            self.chat_raw[idx - 1] = merged_message

            # delete split messages by index after merging
            del self.chat_raw[idx]

    def parse_date_ios(self, line: str) -> datetime.datetime or str:
        """
        Takes single element of chat_raw of an ios chat_raw and returns date.

        Parameters
        ----------
        line : str
            single message (list entry of chat_raw).

        Returns
        -------
        datetime.datetime
            happened datetime.datetime of chat message.

        """
        # split every line of chat between the first brackets
        try:
            date_string = line.split('[')[1].split(']')[0]
            # create datetime obj from remaining date format dd.mm.yy, HH:MM:SS
            message_date = datetime.datetime.strptime(date_string, '%d.%m.%y, %H:%M:%S')  # datetime.datetime
            return message_date
        except ValueError:
            return "unparsable date"
        except IndexError:
            return "unparsable timestamp"

    def parse_date_android(self, line: str) -> datetime.datetime or str:
        """
        Takes single element of chat_raw of an android chat_raw
        and returns its date.

        Parameters
        ----------
        line : str
            single message (list entry of chat_raw).

        Returns
        -------
        datetime.datetime
            returns happened datetime.datetime of chat message.

        """
        # split every line of chat after "-"
        date_string = line.split('-')[0].strip()
        # create datetime obj from remaining date format dd.mm.yy, HH:MM:SS
        try:
            message_date = datetime.datetime.strptime(date_string, '%d.%m.%y, %H:%M')  # datetime.datetime
            return message_date
        except IndexError:
            return "unparsable"

    def get_message_sender_android(self, line: str) -> str:
        """
        Get the sender of a single android chat message

        Parameters
        ----------
        line : str
            single message (list entry of chat_raw).

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

    def get_message_sender_ios(self, line: str) -> str:
        """
        Get the sender of a single ios chat message

        Parameters
        ----------
        line : str
            single message (list entry of chat_raw).

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

    def chop_message_ios(self, line: str) -> str:
        """
        Extract the actual message from a raw ios message.
        Chop timestamp and sender to only get raw text.

        Parameters
        ----------
        line : str
            single message (list entry of chat_raw).

        Returns
        -------
        str
            returns only the actual message.

        """
        # split at 3rd ':', which indicates message start after sender tag
        try:
            return line.split(':', 3)[3].strip()  # str
        except IndexError:
            return "unparsable"

    def chop_message_android(self, line: str) -> str:
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
            return line.split(':', 2)[2].strip()  # type(str)
        except IndexError:
            return "unparsable"

    def get_emojis(self, message: str) -> List[any]:
        """
        Returns a list of all emojis in a message

        Parameters
        ----------
        message : str
            single message (list element of chat_raw).

        Returns
        -------
        List[any]
            returns list of all emojis used in the message.

        """
        return [char for char in message if char in emoji_util.UNICODE_EMOJI['en']]

    def demojize_message(self, message: str) -> str:
        """
        Remove emojis from message text.

        Parameters
        ----------
        message : str
            single message (list element of chat_raw).

        Returns
        -------
        str
            returns a demojized string.

        """
        # find all emojis in message
        emojis_in_message = [char for char in message if char in emoji_util.UNICODE_EMOJI['en']]
        # replace all emojis in message with whitespace
        for emoji in emojis_in_message:
            message = message.replace(emoji, ' ')

        return message

    def annotate_message_types(self,
                               reply_time_threshold:
                               datetime.timedelta = datetime.timedelta(days=2)
                               ):

        """
        Iterate over chat_df and check if message is an reply, 
        follow up or new initiation of the conversation.
        
        ################ EXAMPLE ####################
        # [MSG_A] day1 10:00 sender_a: HI!
        # [MSG B] day1 10:05 sender_b: Hi!
        # --> MSG B is an reply from sender_b
        #
        # [MSG_A] day1 10:00 sender_a: bye!
        # [MSG B] day9 12:05 sender_b: long time no see!
        # --> MSG B is a new initiation of the conversation after a longer period of time
        #
        # [MSG_A] day1 10:00 sender_a: hello!
        # [MSG B] day4 01:05 sender_a: hellooooooo?!
        # --> MSG B is also a new initiation by sender_a
        #
        # [MSG_A] day1 10:00 sender_a: can you bring me something from the store?
        # [MSG B] day1 12:01 sender_a: some milk and icecream!
        # --> MSG B is a follow up by the same sender
        #############################################
        
        Parameters
        ----------
        reply_time_threshold : datetime.timedelta
            Timedelta threshold between two messages indicates a new initiation
            of the conversation.

        """

        for idx, row in self.chat_df.iterrows():
            if idx == 0:
                # first message is always an initiation and has no time diff or reply time
                self.chat_df.at[idx, 'message_type'] = "initiation"
                continue

            # get sender and time_diff between messages
            sender_b = row['sender']
            sender_a = self.chat_df.iloc[idx - 1]['sender']
            time_delta = row['time_diff']  # datetime.timedelta

            # determine whether the message is a reply, follow up or initiation
            if sender_b == sender_a:
                if time_delta < reply_time_threshold:
                    # if sender_a and sender_b are the same and 
                    # time between the two messages is < reply_time_threshold,
                    # then it's a "follow up"
                    self.chat_df.at[idx, 'message_type'] = "follow_up"
                else:
                    # if between messages is > reply_time_threshold, 
                    # then it's a new initiation of the conversation
                    # the recipient didn't respond to the previous message :(
                    self.chat_df.at[idx, 'message_type'] = "initiation"

            if sender_a != sender_b:
                if time_delta < reply_time_threshold:
                    # if sender_a and sender_b are NOT the same and
                    # time between the two messages < reply_time_threshold,
                    # then it's a reply
                    self.chat_df.at[idx, 'message_type'] = "reply"
                    self.chat_df.at[idx, 'reply_time_seconds'] = time_delta.seconds
                else:
                    # if time is > reply_time_threshold then it's a
                    # new initiation (or maybe just a sorry? ¯\_(ツ)_/¯)
                    self.chat_df.at[idx, 'message_type'] = "initiation"

    def is_media(self, message: str) -> bool:
        """
        Returns True if the sent message was an image or video

        Parameters
        ----------
        message : str
            message element of chat_df.

        Returns
        -------
        bool
            Indicator if message was an image

        """
        if self.chat_format == 'ios':
            if any([re.search(pattern, message.lower()) for pattern in
                    self.lang_dict['media_identification_messages_ios']]):
                return True
            else:
                return False

        elif self.chat_format == 'android':
            if re.search("IMG-[0-9]*-WA[0-9]*.jpg", message) is not None:
                return True
            else:
                return False

    def get_ngrams(self, lemmas: List[str], n: int) -> List[Tuple]:
        """
        Generate a list of tuples with all ngrams for a given string.

        Parameters
        ----------
        lemmas : List[str]
            Chat message in spacy doc format.
        n : int
            Number of n for n-grams.

        Returns
        -------
        List[Tuple]
            returns list of tuples with ngrams of the given string.

        """

        n_grams = ngrams(lemmas, n)
        return [n for n in n_grams]

    def annotate_questions(self, spacy_doc: Any) -> bool:
        """
        Returns True if message is a question.

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

    def check_for_url(self, msg: str) -> bool:
        """
        Returns True if the message contains URL

        Parameters
        ----------
        msg : str
            message element of chat_df.

        Returns
        -------
        bool
            Returns True if the message contains URL.

        """

        if "www" in msg or "http" in msg:
            return True
        else:
            return False

    def get_url_domain(self, input_string: str) -> str or None:
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

    def get_lemma(self, token: spacy.tokens.token.Token) -> str:
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
            Lemma if exists. Else returns just the text.

        """

        if token.lemma_:
            return token.lemma_
        else:
            return token.text

    def parse_chat(self):
        start_time = datetime.datetime.now()
        self.load_chat_file()  # List[str]
        self.determine_chat_format()  # str

        if self.chat_format == "android":
            self.check_message_integrity_android()  # List[str]
        elif self.chat_format == "ios":
            self.check_message_integrity_ios()  # List[str]

        self.chat_df = pd.DataFrame(self.chat_raw, columns=['message_raw'])
        print(f"df time: {datetime.datetime.now() - start_time}")

        if self.chat_format == 'ios':
            self.chat_df['sender'] = self.chat_df['message_raw'].apply(lambda x: self.get_message_sender_ios(x))
            self.chat_df['timestamp'] = self.chat_df['message_raw'].apply(lambda x: self.parse_date_ios(x))
            self.chat_df['message'] = self.chat_df['message_raw'].apply(lambda x: self.chop_message_ios(x))
        else:
            self.chat_df['sender'] = self.chat_df['message_raw'].apply(lambda x: self.get_message_sender_android(x))
            self.chat_df['timestamp'] = self.chat_df['message_raw'].apply(lambda x: self.parse_date_android(x))
            self.chat_df['message'] = self.chat_df['message_raw'].apply(lambda x: self.chop_message_android(x))

        # drop messages without valid timestamp (parsing errors)
        self.chat_df = self.chat_df[self.chat_df['timestamp'].apply(lambda x: isinstance(x, datetime.datetime))]
        self.chat_df = self.chat_df.sort_values('timestamp').reset_index(drop=True)

        # further timestamp extractions
        self.chat_df['date'] = self.chat_df['timestamp'].apply(lambda x: x.date())
        self.chat_df['time'] = self.chat_df['timestamp'].apply(lambda x: x.time())
        self.chat_df['weekday'] = self.chat_df['timestamp'].apply(lambda x: x.strftime('%A'))
        self.chat_df['hour'] = self.chat_df['time'].apply(lambda x: x.hour)

        # extract all unique senders
        self.senders = self.chat_df['sender'].unique().tolist()

        # extract emojis
        self.chat_df['emojis'] = self.chat_df['message'].apply(lambda x: self.get_emojis(x))
        self.chat_df['demojized_msg'] = self.chat_df['message'].apply(lambda x: self.demojize_message(x))

        # annotations for message type and reply time
        cache_time = datetime.datetime.now()
        self.chat_df['time_diff'] = self.chat_df['timestamp'] - self.chat_df['timestamp'].shift(periods=1)
        self.annotate_message_types()
        self.chat_df['is_media'] = self.chat_df['message_raw'].apply(lambda x: self.is_media(x))
        # set all messages with images to empty strings
        self.chat_df.loc[self.chat_df['is_media'] == True, ['message', 'demojized_msg']] = ""
        print(f"message annotations time: {datetime.datetime.now() - cache_time}")

        # parse urls and get domains
        cache_time = datetime.datetime.now()
        self.chat_df['has_url'] = self.chat_df['message'].apply(self.check_for_url)
        self.chat_df['url_domain'] = self.chat_df[self.chat_df['has_url'] == True]['message'] \
            .apply(lambda x: self.get_url_domain(x))
        print(f"url parse and annotations time: {datetime.datetime.now() - cache_time}")

        # if nlp is desired start processing here
        if self.nlp:
            # initialize NLP model
            cache_time = datetime.datetime.now()
            # check for language

            try:
                nlp_model = spacy.load(self.lang_dict['nlp_model_name'],
                                       exclude=['senter', 'sentencizer', 'attribute_ruler',
                                                'parser', 'morphologizer', 'ner'])
            except OSError:
                raise OSError(
                    f"{self.lang_dict['nlp_model_name']} spacy language model not found. Make sure to install via "
                    f"'python -m spacy download de_core_news_md'")

            nlp_model.remove_pipe("ner")
            nlp_model.remove_pipe("parser")

            # add stopwords
            with open(self.lang_dict['stopword_file_name'], 'r', encoding='UTF-8') as infile:
                stopwords = infile.read().splitlines()

            for w in stopwords:
                nlp_model.vocab[w].is_stop = True

            # NLP parsing
            self.chat_df['spacy_doc'] = self.chat_df['demojized_msg'].apply(lambda msg: nlp_model(msg.lower()))
            self.chat_df['words'] = self.chat_df['spacy_doc'] \
                .apply(lambda doc: [token.text for token in doc
                                    if token.is_stop is False
                                    and token.is_punct is False
                                    and token.is_bracket is False
                                    and token.is_quote is False
                                    and token.is_space is False])

            self.chat_df['nouns'] = self.chat_df['spacy_doc'] \
                .apply(lambda doc: [token.text for token in doc
                                    if token.pos_ == "NOUN"
                                    and token.is_stop is False
                                    and token.is_punct is False
                                    and token.is_bracket is False
                                    and token.is_quote is False
                                    and token.is_space is False])

            self.chat_df['verbs'] = self.chat_df['spacy_doc'] \
                .apply(lambda doc: [token.text for token in doc
                                    if token.pos_ == "VERB"
                                    and token.is_stop is False
                                    and token.is_punct is False
                                    and token.is_bracket is False
                                    and token.is_quote is False
                                    and token.is_space is False])

            self.chat_df['lemmas'] = self.chat_df['spacy_doc'] \
                .apply(lambda doc: [token.lemma_ for token in doc
                                    if token.is_stop is False
                                    and token.is_punct is False
                                    and token.is_bracket is False
                                    and token.is_quote is False
                                    and token.is_space is False])

            self.chat_df['is_question'] = self.chat_df['spacy_doc'].apply(self.annotate_questions)

            # generate ngrams
            self.chat_df['trigrams'] = self.chat_df['lemmas'].apply(self.get_ngrams, n=3)
            self.chat_df['bigrams'] = self.chat_df['lemmas'].apply(self.get_ngrams, n=2)
            print(f"NLP time: {datetime.datetime.now() - cache_time}")

        print(f"took {datetime.datetime.now() - start_time}")


class UnknownChatFormat(Exception):
    """
    Raise when chat format could not be detected
    """

    def __init__(self, message="could not detect if android or iOS was used"):
        self.message = message
        super().__init__(self.message)
