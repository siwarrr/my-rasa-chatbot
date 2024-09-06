# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these actions:
# https://rasa.com/docs/rasa/custom-actions

# actions.py
from asyncore import dispatcher
import logging
import numpy as np
import openai  # type: ignore
from openai.error import RateLimitError # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore

from pymongo import MongoClient, errors # type: ignore
from typing import Any, Optional, Text, Dict, List, Tuple
from rasa_sdk import Action, Tracker # type: ignore
from rasa_sdk.executor import CollectingDispatcher # type: ignore
from rasa_sdk.events import SlotSet # type: ignore

import spacy
from fuzzywuzzy import fuzz, process
from transformers import pipeline

import re
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from elasticsearch import Elasticsearch, helpers # type: ignore
from bson import ObjectId # type: ignore
from pymongo.errors import OperationFailure # type: ignore
from googletrans import Translator

# Initialiser le pipeline de génération de texte
generator = pipeline('text-generation', model='gpt2')

client = MongoClient('mongodb://localhost:27017/')  
db = client['inclusiveLearning']
courses_collection = db["courses"]
topics_collection = db["topics"]
lessons_collection = db["lessons"]
quizzes_collection = db["quizzes"]
users_collection = db["users"]
index_name = "title_text_description_text"


# Recréer l'index avec le nom correct
try:
    courses_collection.create_index(
        [("title", "text"), ("description", "text")],
        name=index_name
    )
    print(f"Index '{index_name}' created successfully.")
except OperationFailure as e:
    print(f"Error creating index: {e}")

# Initialiser Elasticsearch avec le schéma 'http'
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

def sync_courses_to_elasticsearch():
    courses = list(courses_collection.find())
    actions = [
        {
            "_index": "courses",
            "_id": str(course["_id"]),
            "_source": {
                "title": course["title"],
                "description": course["description"],
                # Ajoutez d'autres champs que vous voulez indexer
            }
        }
        for course in courses
    ]
    helpers.bulk(es, actions)

# Synchroniser les données
sync_courses_to_elasticsearch()

class ActionHelloWorld(Action):

    def name(self) -> str:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="Hello World! How are you!")
        return []

class ActionChatbot(Action):
    def name(self) -> Text:
        return "action_chatbot"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get('text')
        response = f"You said: {user_message}"
        dispatcher.utter_message(text=response)
        return []
class ActionUtterGreet(Action):
    def name(self) -> str:
        return "utter_did_that_help"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        message = tracker.latest_message.get("text")
        dispatcher.utter_message(text=message)
        return []

class ActionProvideChatLink(Action):
    def name(self) -> Text:
        return "action_provide_chat_link"

    def translate_text(self, text: str, target_language: str) -> str:
        translator = Translator()
        translation = translator.translate(text, dest=target_language)
        return translation.text

    def detect_language(self, text: str) -> str:
        translator = Translator()
        detected_lang = translator.detect(text).lang
        return detected_lang

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_id = tracker.sender_id
        user_query = tracker.latest_message.get('text')

        if user_id is None or user_id == "" or user_id == "default":
            login_link = "<a href='http://localhost:4200/welcome/login' data-link-type='login'>click here to log in</a>"
            message = f"You need to log in to our platform first. {login_link}"
            user_language = self.detect_language(user_query)
            if user_language == "ar":
                message_ar = message.replace("You need to log in to our platform first.", "تحتاج إلى تسجيل الدخول إلى منصتنا أولاً.")
                message_ar = message_ar.replace("click here to log in", "انقر هنا لتسجيل الدخول")
                dispatcher.utter_message(text=message_ar)
            elif user_language == "fr":
                message_fr = message.replace("You need to log in to our platform first.", "Vous devez d'abord vous connecter à notre plateforme.")
                message_fr = message_fr.replace("click here to log in", "cliquez ici pour vous connecter")
                dispatcher.utter_message(text=message_fr)
            else:
                dispatcher.utter_message(text=message)
            return []

        chat_link = "<a href='http://localhost:4200/learner/chat' data-link-type='chat'>click here to chat</a>"
        message = f"You can talk with your teacher on the chat page: just click on 'messages' in the header on the right side of the page. {chat_link}"

        user_language = self.detect_language(user_query)
        if user_language == "ar":
            message_ar = message.replace("You can talk with your teacher on the chat page:", "يمكنك التحدث مع معلمك على صفحة الدردشة:")
            message_ar = message_ar.replace("just click on 'messages' in the header on the right side of the page.", "ما عليك سوى النقر على 'الرسائل' في العنوان أعلى الصفحة.")
            message_ar = message_ar.replace("click here to chat", "انقر هنا للدردشة")
            dispatcher.utter_message(text=message_ar)
        elif user_language == "fr":
            message_fr = message.replace("You can talk with your teacher on the chat page:", "Vous pouvez discuter avec votre enseignant sur la page de chat :")
            message_fr = message_fr.replace("just click on 'messages' in the header on the right side of the page.", "il vous suffit de cliquer sur 'messages' dans l'en-tête à droite de la page.")
            message_fr = message_fr.replace("click here to chat", "cliquez ici pour discuter")
            dispatcher.utter_message(text=message_fr)
        else:
            translated_message = self.translate_text(message, user_language)
            dispatcher.utter_message(text=translated_message)

        return []
    
class ActionUtterListCourses(Action):
    def name(self) -> Text:
        return "action_list_courses"

    def detect_language(self, text: str) -> str:
        translator = Translator()
        detected_lang = translator.detect(text).lang
        return detected_lang

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_id = tracker.sender_id
        user_query = tracker.latest_message.get('text')

        if user_id is None or user_id == "" or user_id == "default":
            login_link = "<a href='http://localhost:4200/welcome/login' data-link-type='login'>click here to log in</a>"
            message = f"You need to log in to our platform first. {login_link}"
            user_language = self.detect_language(user_query)
            if user_language == "ar":
                message = message.replace("You need to log in to our platform first.", "تحتاج إلى تسجيل الدخول إلى منصتنا أولاً.")
                message = message.replace("click here to log in", "انقر هنا لتسجيل الدخول")
            elif user_language == "fr":
                message = message.replace("You need to log in to our platform first.", "Vous devez d'abord vous connecter à notre plateforme.")
                message = message.replace("click here to log in", "cliquez ici pour vous connecter")
            dispatcher.utter_message(text=message)
            return []

        primary_link = "<a href='http://localhost:4200/learner/course-space/65f0dcccc96fff4befcdf508'>Click here to view the primary space</a>"
        high_link = "<a href='http://localhost:4200/learner/course-space/65f0dcf8c96fff4befcdf50a'>Click here to view the high school space</a>"
        univ_link = "<a href='http://localhost:4200/learner/course-space/65f0dd19c96fff4befcdf50c'>Click here to view the university space</a>"

        message = (f"Here is the list of courses for each course space on our platform:<br>"
                   f"Primary space: {primary_link}<br>"
                   f"High school space: {high_link}<br>"
                   f"University space: {univ_link}")

        user_language = self.detect_language(user_query)
        if user_language == "ar":
            message = message.replace("Here is the list of courses for each course space on our platform:", "إليك قائمة الدورات لكل مساحة دراسية على منصتنا:")
            message = message.replace("Primary space", "المرحلة الابتدائية")
            message = message.replace("High school space", "المرحلة الثانوية")
            message = message.replace("University space", "المرحلة الجامعية")
            message = message.replace("Click here to view the primary space", "انقر هنا لعرض مساحة المرحلة الابتدائية")
            message = message.replace("Click here to view the high school space", "انقر هنا لعرض مساحة المرحلة الثانوية")
            message = message.replace("Click here to view the university space", "انقر هنا لعرض مساحة المرحلة الجامعية")

        elif user_language == "fr":
            message = message.replace("Here is the list of courses for each course space on our platform:", "Voici la liste des cours pour chaque espace de cours sur notre plateforme :")
            message = message.replace("Primary space", "Espace primaire")
            message = message.replace("High school space", "Espace secondaire")
            message = message.replace("University space", "Espace universitaire")
            message = message.replace("Click here to view the primary space", "Cliquez ici pour voir l'espace primaire")
            message = message.replace("Click here to view the high school space", "Cliquez ici pour voir l'espace secondaire")
            message = message.replace("Click here to view the university space", "Cliquez ici pour voir l'espace universitaire")

        dispatcher.utter_message(text=message)
        return []

class ActionProvideLoginLink(Action):
    def name(self) -> Text:
        return "action_provide_login_link"

    def translate_text(self, text: str, target_language: str) -> str:
        translator = Translator()
        try:
            translation = translator.translate(text, dest=target_language)
            return translation.text
        except Exception as e:
            logging.error(f"Error translating text: {e}")
            return text  # Retourne le texte d'origine en cas d'erreur

    def detect_language(self, text: str) -> str:
        translator = Translator()
        try:
            detected_lang = translator.detect(text).lang
            return detected_lang
        except Exception as e:
            logging.error(f"Error detecting language: {e}")
            return 'en'  # Définit l'anglais comme langue par défaut en cas d'erreur

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_query = tracker.latest_message.get('text')
        user_language = self.detect_language(user_query)
        
        login_link = "<a href='http://localhost:4200/welcome/login'>click here</a>"
        message = f"To log in to our platform, simply enter your email and password, then click the 'login' button. {login_link}"

        if user_language == "ar":
            message = message.replace("To log in to our platform, simply enter your email and password, then click the 'login' button.", "لتسجيل الدخول إلى منصتنا، ما عليك سوى إدخال بريدك الإلكتروني وكلمة المرور، ثم النقر فوق زر 'تسجيل الدخول'.")
            message = message.replace("click here", "انقر هنا")
        elif user_language == "fr":
            message = message.replace("To log in to our platform, simply enter your email and password, then click the 'login' button.", "Pour vous connecter à notre plateforme, entrez simplement votre email et votre mot de passe, puis cliquez sur le bouton 'login'.")
            message = message.replace("click here", "cliquez ici")
        else:
            message = self.translate_text(message, user_language)

        dispatcher.utter_message(text=message)
        return []

class ActionProvideSignupLink(Action):
    def name(self) -> Text:
        return "action_provide_signup_link"

    def translate_text(self, text: str, target_language: str) -> str:
        translator = Translator()
        try:
            translation = translator.translate(text, dest=target_language)
            return translation.text
        except Exception as e:
            logging.error(f"Error translating text: {e}")
            return text  # Retourne le texte d'origine en cas d'erreur

    def detect_language(self, text: str) -> str:
        translator = Translator()
        try:
            detected_lang = translator.detect(text).lang
            return detected_lang
        except Exception as e:
            logging.error(f"Error detecting language: {e}")
            return 'en'  # Définit l'anglais comme langue par défaut en cas d'erreur

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_query = tracker.latest_message.get('text')
        user_language = self.detect_language(user_query)

        signup_link = "<a href='http://localhost:4200/welcome/register'>click here</a>"
        message = ("To register, you need to enter your full name, email, choose whether you want to register as:<br>"
                   "1st choice: 'Learner',<br>2nd choice: 'Teacher'.<br>"
                   "Then select if you have your type of specific need from the dropdown list,<br>"
                   "enter a password,<br>re-enter this password to confirm it,<br>"
                   f"and then click the 'Register' button.<br>{signup_link}")

        if user_language == "ar":
            message = message.replace(
                "To register, you need to enter your full name, email, choose whether you want to register as:<br>1st choice: 'Learner',<br>2nd choice: 'Teacher'.<br>Then select if you have your type of specific need from the dropdown list,<br>enter a password,<br>re-enter this password to confirm it,<br>and then click the 'Register' button.<br>",
                "للتسجيل، تحتاج إلى إدخال اسمك الكامل، البريد الإلكتروني، اختر ما إذا كنت ترغب في التسجيل كـ: الخيار الأول: 'متعلم'، الخيار الثاني: 'معلم'.<br>ثم اختر نوع احتياجك الخاص من القائمة المنسدلة،<br>أدخل كلمة مرور،<br>أعد إدخال كلمة المرور هذه لتأكيدها،<br>ثم انقر فوق زر 'تسجيل'.<br>"
            )
            message = message.replace("click here", "انقر هنا")
        elif user_language == "fr":
            message = message.replace(
                "To register, you need to enter your full name, email, choose whether you want to register as:<br>1st choice: 'Learner',<br>2nd choice: 'Teacher'.<br>Then select if you have your type of specific need from the dropdown list,<br>enter a password,<br>re-enter this password to confirm it,<br>and then click the 'Register' button.<br>",
                "Pour vous inscrire, vous devez entrer votre nom complet, votre email, choisir si vous voulez vous inscrire en tant que:<br>1er choix: 'Apprenant',<br>2ème choix: 'Enseignant'.<br>Puis sélectionnez si vous avez votre type de besoin spécifique dans la liste déroulante,<br>entrez un mot de passe,<br>saisissez à nouveau ce mot de passe pour le confirmer,<br>puis cliquez sur le bouton 'S'inscrire'.<br>"
            )
            message = message.replace("click here", "cliquez ici")
        else:
            message = self.translate_text(message, user_language)

        dispatcher.utter_message(text=message)
        return []

class ActionRedirectToLogin(Action):

    def name(self) -> str:
        return "action_redirect_to_login"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(json_message={"redirect": "http://localhost:4200/welcome/login"})
        return []
    
def get_embedding(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except RateLimitError:
        return "Sorry, I've exceeded my API quota. Please try again later."

# Fonction pour prétraiter le texte
def preprocess_text(text_list):
    preprocessed_texts = []
    for text in text_list:
        # Convertir en minuscules
        text = text.lower()
        preprocessed_texts.append(text)
    return preprocessed_texts

# Fonction pour calculer la matrice TF-IDF
def calculate_tfidf(texts):
    # Initialiser le vectoriseur TF-IDF avec des paramètres appropriés
    tfidf_vectorizer = TfidfVectorizer()
    # Calculer la matrice TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_matrix, tfidf_vectorizer

# Fonction pour calculer la similarité cosine
def calculate_similarity(query_vector, course_vectors):
    # Vérifier les dimensions
    if query_vector.shape[1] != course_vectors.shape[1]:  
        raise ValueError("Les dimensions des vecteurs de requête et de cours ne correspondent pas.")
    
    # Calculer la similarité cosine
    similarities = cosine_similarity(query_vector, course_vectors)
    return similarities

# Fonction pour indexer les cours depuis MongoDB vers Elasticsearch
def search_courses(query):
    response = es.search(
        index="courses",
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "description"]
                }
            }
        }
    )
    search_results = [
        {
            "course_id": hit['_id'],
            "title": hit['_source']['title'],
            "description": hit['_source']['description']
        }
        for hit in response['hits']['hits']
    ]
    return search_results

class ActionSearchCourses(Action):
    def name(self) -> Text:
        return "action_search_courses"

    def translate_text(self, text: str, target_language: str) -> str:
        translator = Translator()
        translation = translator.translate(text, dest=target_language)
        return translation.text

    def detect_language(self, text: str) -> str:
        translator = Translator()
        detected_lang = translator.detect(text).lang
        return detected_lang

    def extract_course(self, user_query: str) -> Optional[str]:
        course_match = re.search(r'(?:learn about|courses on|information on|classes on|to study|know about|courses related to|interested in learning about)\s+(.+)', user_query, re.IGNORECASE)
        if not course_match:
            course_match = re.search(r'(?:apprendre sur|cours sur|des courses sur|informations sur|classes sur|savoir sur|étudier|cours liés à|intéressé par l\'apprentissage de)\s+(.+)', user_query, re.IGNORECASE)
        if not course_match:
            course_match = re.search(r'(?:أريد أن أتعلم عن|دروس حول|معلومات عن|فصول عن|معرفة عن|دورات تتعلق بـ|مهتم بتعلم عن)\s+(.+)', user_query, re.IGNORECASE)

        if course_match:
            return course_match.group(1).strip()
        return None

    def check_user_enrollment(self, user_id: str, course_id: str) -> bool:
        try:
            course = courses_collection.find_one({"_id": ObjectId(course_id)})
            if course and "students" in course:
                return user_id in [str(student_id) for student_id in course["students"]]
        except Exception as e:
            logger.error(f"Error checking enrollment: {str(e)}")
        return False

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_query = tracker.latest_message.get('text')
        user_id = tracker.sender_id

        course = self.extract_course(user_query)
        query_without_course = user_query.replace(course, "").strip() if course else user_query
        user_language = self.detect_language(query_without_course)

        print(f"User ID: {user_id}")
        print(f"Latest Message: {tracker.latest_message}")
        print(f"Detected Language: {user_language}")

        if not user_id or user_id == "default":
            login_link = "<a href='http://localhost:4200/welcome/login' data-link-type='login'>click here to log in</a>"
            message = {
                'en': f"You need to log in to our platform first. {login_link}",
                'ar': "تحتاج إلى تسجيل الدخول إلى منصتنا أولاً. <a href='http://localhost:4200/welcome/login' data-link-type='login'>انقر هنا لتسجيل الدخول</a>",
                'fr': "Vous devez d'abord vous connecter à notre plateforme. <a href='http://localhost:4200/welcome/login' data-link-type='login'>cliquez ici pour vous connecter</a>"
            }.get(user_language, f"You need to log in to our platform first. {login_link}")
            dispatcher.utter_message(text=message)
            return []

        if course:
            search_results = search_courses(course)
            if search_results:
                intro_messages = {
                    'ar': "إليك بعض الدورات ذات الصلة:<br>",
                    'fr': "Voici quelques cours pertinents:<br>",
                    'en': "Here are some relevant courses:<br>"
                }
                dispatcher.utter_message(text=intro_messages.get(user_language, "Here are some relevant courses:<br>"))
                course_messages = []
                for idx, course in enumerate(search_results, 1):
                    course_id = course['course_id']
                    course_link = f"http://localhost:4200/learner/course-content/{course_id}"
                    enrolled = self.check_user_enrollment(user_id, course_id)

                    if enrolled:
                        course_message = {
                            'ar': f"الدورة {idx}:<br>Title: {course['title']}<br>Description: {course['description']}<br>لعرض هذه الدورة <a href='{course_link}' data-link-type='course' data-enrolled='true' data-course-id='{course_id}'>click here</a>.<br>لقد سجلت في هذه الدورة.",
                            'fr': f"Cours {idx}:<br>Title: {course['title']}<br>Description: {course['description']}<br>Pour voir ce cours <a href='{course_link}' data-link-type='course' data-enrolled='true' data-course-id='{course_id}'>click here</a>.<br>Vous êtes inscrit à ce cours.",
                            'en': f"Course {idx}:<br>Title: {course['title']}<br>Description: {course['description']}<br>To view this course <a href='{course_link}' data-link-type='course' data-enrolled='true' data-course-id='{course_id}'>click here</a>.<br>You are enrolled in this course."
                        }
                    else:
                        course_message = {
                            'ar': f"الدورة {idx}:<br>Title: {course['title']}<br>Description: {course['description']}<br>لعرض هذه الدورة <a href='{course_link}' data-link-type='course' data-enrolled='false' data-course-id='{course_id}'>click here</a>.<br>لم تسجل في هذه الدورة. يرجى التسجيل للوصول إلى المحتوى.",
                            'fr': f"Cours {idx}:<br>Title: {course['title']}<br>Description: {course['description']}<br>Pour voir ce cours <a href='{course_link}' data-link-type='course' data-enrolled='false' data-course-id='{course_id}'>click here</a>.<br>Vous n'êtes pas inscrit à ce cours. Veuillez vous inscrire pour accéder au contenu.",
                            'en': f"Course {idx}:<br>Title: {course['title']}<br>Description: {course['description']}<br>To view this course <a href='{course_link}' data-link-type='course' data-enrolled='false' data-course-id='{course_id}'>click here</a>.<br>You are not enrolled in this course. Please enroll to access the content."
                        }
                    course_messages.append(course_message.get(user_language, course_message['en']))
                dispatcher.utter_message(text="<br>".join(course_messages))
                return [SlotSet("search_results", json.dumps(search_results)), SlotSet("last_query", user_query)]
            else:
                no_courses_message = {
                    'ar': "لم يتم العثور على دورات للكلمة الرئيسية المعطاة.",
                    'fr': "Aucun cours trouvé pour le mot-clé donné.",
                    'en': "No courses found for the given keyword."
                }
                dispatcher.utter_message(text=no_courses_message.get(user_language, "No courses found for the given keyword."))
                return [SlotSet("search_results", None), SlotSet("last_query", user_query)]
        else:
            no_course_message = "Sorry, I couldn't understand the course you want to learn about."
            if user_language == "ar":
                no_course_message = "عذرًا، لم أتمكن من فهم الدورة التي تريد تعلمها."
            elif user_language == "fr":
                no_course_message = "Désolé, je n'ai pas compris le cours que vous souhaitez apprendre."
            elif user_language != "en":
                no_course_message = self.translate_text(no_course_message, user_language)

            dispatcher.utter_message(text=no_course_message)
            return []

    
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ActionAnswerCourseContentQuery(Action):
    def name(self) -> Text:
        return "action_answer_course_content_query"

    def translate_text(self, text: str, target_language: str) -> str:
        translator = Translator()
        translation = translator.translate(text, dest=target_language)
        return translation.text

    def detect_language(self, text: str) -> str:
        translator = Translator()
        detected_lang = translator.detect(text).lang
        return detected_lang

    def get_course_link(self, course_id: str) -> str:
        base_url = "http://localhost:4200/learner/course-content/"
        return f"{base_url}{course_id}"

    def extract_subject(self, user_query: str) -> Optional[str]:
        patterns = [
            r'\b(?:understand|explain|learn about|teach me|tell me about|information about|how does|what is|can you explain)\s+(.+)',
            r'\b(?:comprendre|expliquer|apprendre sur|enseigner|parlez-moi de|informations sur|comment fonctionne|qu\'est-ce que|pouvez-vous expliquer)\s+(.+)',
            r'\b(?:أريد أن أفهم|هل يمكنك شرح|أحتاج لمساعدتك في فهم|أريد أن أتعلم عن|أخبرني عن|اعطني معلومات عن|كيف يعمل|ما هو|هل يمكنك تعليمي)\s+(.+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_query = tracker.latest_message.get('text')
        user_id = tracker.sender_id

        # Extract subject from user query
        subject = self.extract_subject(user_query)
        # Remove the extracted subject from the query for language detection
        query_without_subject = user_query.replace(subject, "").strip() if subject else user_query
        user_language = self.detect_language(query_without_subject)

        print(f"User ID: {user_id}")
        print(f"Latest Message: {tracker.latest_message}")
        print(f"Detected Language: {user_language}")

        # Message for users who are not logged in
        if user_id is None or user_id == "" or user_id == "default":
            login_link = "<a href='http://localhost:4200/welcome/login' data-link-type='login'>click here to log in</a>"
            message = f"You need to log in to our platform first. {login_link}"
            if user_language == "ar":
                message = message.replace("You need to log in to our platform first.", "تحتاج إلى تسجيل الدخول إلى منصتنا أولاً.")
                message = message.replace("click here to log in", "انقر هنا لتسجيل الدخول")
            elif user_language == "fr":
                message = message.replace("You need to log in to our platform first.", "Vous devez d'abord vous connecter à notre plateforme.")
                message = message.replace("click here to log in", "cliquez ici pour vous connecter")
            dispatcher.utter_message(text=message)
            return []

        if subject:
            print(f"Extracted subject: {subject}")

            lessons = self.get_all_courses_content(subject)
            if lessons:
                response = f"Here are the lessons related to '{subject}':\n"
                for idx, lesson in enumerate(lessons):
                    course_id = lesson['course_id']
                    course_title = lesson['course_title']
                    lesson_title = lesson['lesson_title']
                    enrolled = self.check_user_enrollment(user_id, course_id)
                    course_link = self.get_course_link(course_id)

                    response += f"{idx + 1}. <b>Course:</b> {course_title} - <b>Lesson:</b> {lesson_title}<br>\n"
                    if enrolled:
                        response += f"To consult this course for better understanding <a href={course_link}>click here</a>.<br>\nYou are enrolled in this course.<br>\n"
                    else:
                        response += f"To consult this course for better understanding <a href={course_link}>click here</a>.<br>\nYou are not enrolled in this course. Please enroll to access the content.<br>\n"

                if user_language == "ar":
                    response = response.replace("Here are the lessons related to", "إليك الدروس المتعلقة بـ")
                    response = response.replace("Course:", "الدورة:")
                    response = response.replace("Lesson:", "الدرس:")
                    response = response.replace("To consult this course for better understanding", "للاطلاع على هذه الدورة لفهم أفضل")
                    response = response.replace("You are enrolled in this course.", "لقد قمت بالتسجيل في هذه الدورة.")
                    response = response.replace("You are not enrolled in this course. Please enroll to access the content.", "لم تقم بالتسجيل في هذه الدورة. يرجى التسجيل للوصول إلى المحتوى.")
                elif user_language == "fr":
                    response = response.replace("Here are the lessons related to", "Voici les leçons liées à")
                    response = response.replace("Course:", "Cours :")
                    response = response.replace("Lesson:", "Leçon :")
                    response = response.replace("To consult this course for better understanding", "Pour consulter ce cours pour une meilleure compréhension")
                    response = response.replace("You are enrolled in this course.", "Vous êtes inscrit à ce cours.")
                    response = response.replace("You are not enrolled in this course. Please enroll to access the content.", "Vous n'êtes pas inscrit à ce cours. Veuillez vous inscrire pour accéder au contenu.")
                elif user_language != "en":
                    response = self.translate_text(response, user_language)

                dispatcher.utter_message(text=response)
                return [SlotSet("lessons", lessons), SlotSet("subject", subject)]
            else:
                logger.debug(f"No lessons found for the subject: {subject}")
                no_lessons_message = f"No lessons found for '{subject}' in the available courses."
                if user_language == "ar":
                    no_lessons_message = f"لم يتم العثور على دروس لـ '{subject}' في الدورات المتاحة."
                elif user_language == "fr":
                    no_lessons_message = f"Aucune leçon trouvée pour '{subject}' dans les cours disponibles."
                elif user_language != "en":
                    no_lessons_message = self.translate_text(no_lessons_message, user_language)

                dispatcher.utter_message(text=no_lessons_message)
                return []
        else:
            logger.debug(f"No subject found in the user query: {user_query}")
            no_subject_message = "Sorry, I couldn't understand the subject you want to learn about."
            if user_language == "ar":
                no_subject_message = "عذرًا، لم أتمكن من فهم الموضوع الذي تريد تعلمه."
            elif user_language == "fr":
                no_subject_message = "Désolé, je n'ai pas compris le sujet que vous souhaitez apprendre."
            elif user_language != "en":
                no_subject_message = self.translate_text(no_subject_message, user_language)

            dispatcher.utter_message(text=no_subject_message)
            return []

    def get_all_courses_content(self, topic: str) -> List[Dict[Text, Any]]:
        lessons = []
        logger.debug(f"Searching for topic: {topic}")

        try:
            if not lessons_collection.count_documents({}):
                logger.debug("No lessons found in the database.")
                return lessons

            for lesson in lessons_collection.find():
                lesson_name = lesson.get("name", "").lower()
                lesson_content = lesson.get("content", "").lower()
                if topic.lower() in lesson_name or topic.lower() in lesson_content:
                    # Fetch the topic containing this lesson
                    topic_data = topics_collection.find_one({"lessons": lesson["_id"]})
                    if topic_data:
                        # Fetch the course containing this topic
                        course_data = courses_collection.find_one({"sections": topic_data["_id"]})
                        if course_data:
                            lesson_dict = {
                                "course_id": str(course_data["_id"]),
                                "course_title": course_data["title"],
                                "lesson_title": lesson.get("name", "No content"),
                                "video_url": lesson.get("videoUrl", "No video")
                            }
                            lessons.append(lesson_dict)

            logger.debug(f"Total lessons found: {len(lessons)}")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")

        return lessons

    def check_user_enrollment(self, user_id: str, course_id: str) -> bool:
        try:
            course = courses_collection.find_one({"_id": ObjectId(course_id)})
            if course and "students" in course:
                return user_id in [str(student_id) for student_id in course["students"]]
        except Exception as e:
            logger.error(f"Error checking enrollment: {str(e)}")
        return False

class ActionListEnrolledCourses(Action):
    def name(self) -> Text:
        return "action_list_enrolled_courses"

    def get_user_fullname(self, user_id: str) -> str:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if user and "fullname" in user:
            return user["fullname"]
        return "User"

    def get_enrolled_courses(self, user_id: str) -> List[Dict[Text, Any]]:
        courses = list(courses_collection.find({"students": ObjectId(user_id)}))
        return [
            {
                "course_id": str(course.get("_id")), 
                "title": course.get("title", "No title")
            }
            for course in courses
        ]

    def translate_text(self, text: str, target_language: str) -> str:
        translator = Translator()
        translation = translator.translate(text, dest=target_language)
        return translation.text

    def detect_language(self, text: str) -> str:
        translator = Translator()
        detected_lang = translator.detect(text).lang
        return detected_lang

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_id = tracker.sender_id
        user_query = tracker.latest_message.get('text')

        if user_id is None or user_id == "" or user_id == "default":
            login_link = "<a href='http://localhost:4200/welcome/login' data-link-type='login'>click here to log in</a>"
            message = f"You need to log in to our platform first. {login_link}"
            user_language = self.detect_language(user_query)
            if user_language == "ar":
                message = message.replace("You need to log in to our platform first.", "تحتاج إلى تسجيل الدخول إلى منصتنا أولاً.")
                message = message.replace("click here to log in", "انقر هنا لتسجيل الدخول")
            elif user_language == "fr":
                message = message.replace("You need to log in to our platform first.", "Vous devez d'abord vous connecter à notre plateforme.")
                message = message.replace("click here to log in", "cliquez ici pour vous connecter")
            dispatcher.utter_message(text=message)
            return []

        user_fullname = self.get_user_fullname(user_id)
        enrolled_courses = self.get_enrolled_courses(user_id)
        user_language = self.detect_language(user_query)

        if enrolled_courses:
            response = f"Hi {user_fullname}, Here are the courses you are enrolled in:<br>"
            for idx, course in enumerate(enrolled_courses, 1):
                course_link = f"http://localhost:4200/learner/course-content/{course['course_id']}"
                response += f"{idx}. <a href='{course_link}'>{course['title']}</a><br>"
        else:
            response = f"Hi {user_fullname}, you are not enrolled in any courses."

        if user_language == "ar":
            response_ar = response.replace("Hi ", "مرحبا ")
            response_ar = response_ar.replace("Here are the courses you are enrolled in:", "إليك الدورات التي سجلت فيها:")
            response_ar = response_ar.replace("you are not enrolled in any courses.", "لم تسجل في أي دورات.")
            dispatcher.utter_message(text=response_ar)
        elif user_language == "fr":
            response_fr = response.replace("Hi ", "Bonjour ")
            response_fr = response_fr.replace("Here are the courses you are enrolled in:", "Voici les cours auxquels vous êtes inscrit :")
            response_fr = response_fr.replace("you are not enrolled in any courses.", "vous n'êtes inscrit à aucun cours.")
            dispatcher.utter_message(text=response_fr)
        else:
            # Handle other languages or fallback
            translated_response = self.translate_text(response, user_language)
            dispatcher.utter_message(text=translated_response)

        return []

class ActionInquireCourseQuizzes(Action):
    def name(self) -> Text:
        return "action_inquire_course_quizzes"

    def get_course_by_name(self, course_name: str) -> Dict[Text, Any]:
        # Replace with your database call
        return courses_collection.find_one({"title": {"$regex": course_name, "$options": "i"}})

    def get_quizzes(self, section_ids: List[str]) -> List[Dict[Text, Any]]:
        quizzes = []
        for section_id in section_ids:
            section = topics_collection.find_one({"_id": ObjectId(section_id)})
            if section and "quizzes" in section:
                for quiz_id in section["quizzes"]:
                    quiz = quizzes_collection.find_one({"_id": ObjectId(quiz_id)})
                    if quiz:
                        quizzes.append({
                            "quiz_id": str(quiz.get("_id")),
                            "name": quiz.get("name", "No title")
                        })
        return quizzes

    def translate_text(self, text: str, target_language: str) -> str:
        translator = Translator()
        translation = translator.translate(text, dest=target_language)
        return translation.text

    def detect_language(self, text: str) -> str:
        translator = Translator()
        detected_lang = translator.detect(text).lang
        return detected_lang

    def extract_course_name(self, text: str, language: str) -> str:
        course_name = None
        if language == "en":
            # Match everything after "course" until the end or a punctiation
            match = re.search(r'course\s+(.+)', text, re.IGNORECASE)
        elif language == "fr":
            match = re.search(r'cours\s+(.+)', text, re.IGNORECASE)
        elif language == "ar":
            match = re.search(r'دورة\s+(.+)', text, re.IGNORECASE)
        else:
            return None

        if match:
            course_name = match.group(1).strip()
            # Remove any trailing punctuations or connector words if necessary
            course_name = re.sub(r'(contains|des|الاختبارات|المرتبطة|\s|[,.!?])$', '', course_name)
        return course_name

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_query = tracker.latest_message.get('text')
        user_id = tracker.sender_id
        print(f"User ID: {user_id}")
        print(f"Latest Message: {tracker.latest_message}")

        if user_id is None or user_id == "" or user_id == "default":
            login_link = "<a href='http://localhost:4200/welcome/login' data-link-type='login'>click here to log in</a>"
            message = f"You need to log in to our platform first. {login_link}"
            user_language = self.detect_language(user_query)
            if user_language == "ar":
                message = message.replace("You need to log in to our platform first.", "تحتاج إلى تسجيل الدخول إلى منصتنا أولاً.")
                message = message.replace("click here to log in", "انقر هنا لتسجيل الدخول")
            elif user_language == "fr":
                message = message.replace("You need to log in to our platform first.", "Vous devez d'abord vous connecter à notre plateforme.")
                message = message.replace("click here to log in", "cliquez ici pour vous connecter")
            dispatcher.utter_message(text=message)
            return []

        user_language = self.detect_language(user_query)
        course_name = self.extract_course_name(user_query, user_language)

        if not course_name:
            response = "Please provide the course name."
            translated_response = self.translate_text(response, user_language)
            dispatcher.utter_message(text=translated_response)
            return []

        print(f"Extracted Course Name: {course_name}")

        course = self.get_course_by_name(course_name)
        if not course:
            response = f"No course found with the name {course_name}."
            translated_response = self.translate_text(response, user_language)
            dispatcher.utter_message(text=translated_response)
            return []

        course_link = f"http://localhost:4200/learner/course-content/{course['_id']}"

        # Default response if the user is not enrolled
        response = (f"You need to enroll in the course '{course_name}' to access the quizzes.<br>\n"
                    f"<a href='{course_link}' class='course-link' data-course-id='{course['_id']}' data-enrolled='false'>click here</a> to participate")

        if str(user_id) not in [str(student_id) for student_id in course.get("students", [])]:
            # Handle response for users not enrolled
            if user_language == "ar":
                general_response_ar = response.replace(f"You need to enroll in the course '{course_name}' to access the quizzes.<br>\n", 
                                                        f"تحتاج إلى التسجيل في الدورة التدريبية {course_name} للوصول إلى الاختبارات.<br>\n")
                general_response_ar = general_response_ar.replace("click here <a href='", "انقر هنا <a href='")
                general_response_ar = general_response_ar.replace("to participate</a>.", "للمشاركة</a>.")
                dispatcher.utter_message(text=general_response_ar)
            elif user_language == "fr":
                general_response_fr = response.replace(f"You need to enroll in the course '{course_name}' to access the quizzes.<br>\n", 
                                                        f"Vous devez vous inscrire au cours '{course_name}' pour accéder aux quiz.<br>\n")
                general_response_fr = general_response_fr.replace("click here <a href='", "cliquez ici <a href='")
                general_response_fr = general_response_fr.replace("to participate</a>.", "pour participer</a>.")
                dispatcher.utter_message(text=general_response_fr)
            else:
                translated_response = self.translate_text(response, user_language)
                dispatcher.utter_message(text=translated_response)
            return []

        # Define general response
        section_ids = course.get("sections", [])
        quizzes = self.get_quizzes(section_ids)

        if quizzes:
            general_response = f"The course {course_name} contains the following quizzes:<br>\n"
            for idx, quiz in enumerate(quizzes, 1):
                general_response += f"{idx}. {quiz['name']}<br>\n"
            general_response += f"\nTo access the course, <a href='{course_link}' class='course-link' data-enrolled='true'>click here to consult the course</a>."
        else:
            general_response = (f"The course {course_name} does not contain any quizzes.<br>\n"
                                f"<a href='{course_link}' class='course-link' data-enrolled='true'>click here to consult the course</a>.")

        if user_language == "ar":
            # Correctly replace the course name and handle link in Arabic
            general_response_ar = general_response.replace(f"The course {course_name} contains the following quizzes:<br>\n", 
                                                            f"تحتوي الدورة التدريبية {course_name} على الاختبارات التالية:<br>\n")
            general_response_ar = general_response_ar.replace(f"The course {course_name} does not contain any quizzes.<br>\n", 
                                                              f"الدورة التدريبية {course_name} لا تحتوي على أي اختبارات.<br>\n")
            general_response_ar = general_response_ar.replace("To access the course, <a href='", 
                                                              "للوصول إلى الدورة، <a href='")
            general_response_ar = general_response_ar.replace("click here to consult the course</a>.", 
                                                              "انقر هنا لعرض الدورة</a>.")
            dispatcher.utter_message(text=general_response_ar)

        elif user_language == "fr":
            # Correctly replace the course name and handle link in French
            general_response_fr = general_response.replace(f"The course {course_name} contains the following quizzes:<br>\n", 
                                                            f"Le cours {course_name} contient les quiz suivants :<br>\n")
            general_response_fr = general_response_fr.replace(f"The course {course_name} does not contain any quizzes.<br>\n", 
                                                              f"Le cours {course_name} ne contient aucun quiz.<br>\n")
            general_response_fr = general_response_fr.replace("To access the course, <a href='", 
                                                              "Pour accéder au cours, <a href='")
            general_response_fr = general_response_fr.replace("click here to consult the course</a>.", 
                                                              "cliquez ici pour consulter le cours</a>.")
            dispatcher.utter_message(text=general_response_fr)
        else:
            translated_general_response = self.translate_text(general_response, user_language)
            dispatcher.utter_message(text=translated_general_response)

        return [SlotSet("course_name", course_name)]

    
class ActionGenerateAdvancedGPTResponse(Action):
    def name(self) -> Text:
        return "action_generate_advanced_gpt_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_query = tracker.latest_message.get('text')
        user_id = tracker.sender_id
        
        if user_id is None or user_id == "" or user_id == "default":
            login_link = "<a href='http://localhost:4200/welcome/login' data-link-type='login'>click here to log in</a>"
            message = f"You need to log in to our platform first. {login_link}"
            dispatcher.utter_message(text=message)
            return []

        subject_match = re.search(r'\b(?:understand|explain|learn about|teach me|tell me about|information about|how does|what is|can you explain)\s+(.+)', user_query, re.IGNORECASE)
        if subject_match:
            subject = subject_match.group(1).strip()
            lessons = self.get_all_courses_content(subject)

            if lessons:
                prompt = f"I have the following lessons about '{subject}':\n"
                for lesson in lessons:
                    prompt += f"Course: {lesson['course_title']}, Lesson: {lesson['lesson_title']}\n"
                prompt += f"Based on these lessons, can you explain the key concepts about {subject}?"

                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=200
                )
                generated_text = response.choices[0].text.strip()
                dispatcher.utter_message(text=generated_text)
                return [SlotSet("subject", subject)]
            else:
                dispatcher.utter_message(text=f"No lessons found for '{subject}' in the available courses.")
                return []
        else:
            dispatcher.utter_message(text="Sorry, I couldn't understand the subject you want to learn about.")
            return []

    def get_all_courses_content(self, topic: str) -> List[Dict[Text, Any]]:
        lessons = []
        courses = list(courses_collection.find())

        for course in courses:
            if "sections" not in course or not isinstance(course["sections"], list):
                continue

            for section_id in course["sections"]:
                topics = list(topics_collection.find({"_id": ObjectId(section_id)}))
                for topic_data in topics:
                    if topic.lower() in topic_data["name"].lower():
                        lesson_ids = topic_data.get("lessons", [])
                        for lesson_id in lesson_ids:
                            lesson = lessons_collection.find_one({"_id": ObjectId(lesson_id)})
                            if lesson:
                                lesson_dict = {
                                    "course_id": str(course["_id"]),
                                    "course_title": course["title"],
                                    "lesson_title": lesson["name"],
                                    "video_url": lesson.get("videoUrl", "No video")
                                }
                                lessons.append(lesson_dict)
        return lessons