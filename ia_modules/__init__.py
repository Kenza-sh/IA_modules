import azure.functions as func
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import json
from datetime import datetime, timedelta
import dateparser
from babel.dates import format_date
import calendar
from typing import Optional, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class InformationExtractor:
    def __init__(self):
        # Initialisation unique du modèle NER
        logger.info("Initialisation du modèle NER...")
        self.tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
        self.model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
        self.nlp = pipeline('ner', model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        logger.info("Modèle NER initialisé avec succès.")
    def check_noun(self, msg_2_check):
        logger.debug(f"Vérification du nom : {msg_2_check}")
        def check_str(msg_2_check: str) -> bool:
            return isinstance(msg_2_check, str) and bool(msg_2_check.strip()) and any(ele in msg_2_check for ele in ["a", "e", "i", "o", "u", "y"])
        if not check_str(msg_2_check):
            logger.warning(f"Le message {msg_2_check} n'est pas une chaîne valide.")
            return False
        if not re.match(r"^[a-zA-ZÀ-ÿ' -]+$", msg_2_check):
            logger.warning(f"Le message {msg_2_check} contient des caractères invalides.")
            return False
        return True
    def extraire_nom(self, texte):
        logger.info(f"Extraction du nom à partir du texte : {texte}")
        entities = self.nlp(texte)
        for ent in entities:
            if ent['entity_group'] == "PER":
                if self.check_noun(ent['word'].lower()):
                    logger.info(f"Nom extrait : {ent['word'].upper()}")
                    return ent['word'].upper()
        logger.warning("Aucun nom n'a été extrait.")
        return None
    def extraire_prenom(self, texte):
        logger.info(f"Extraction du prénom à partir du texte : {texte}")
        entities = self.nlp(texte)
        for ent in entities:
            if ent['entity_group'] == "PER":
                if self.check_noun(ent['word']):
                    logger.info(f"Prénom extrait : {ent['word']}")
                    return ent['word'].upper()
        logger.warning("Aucun prénom n'a été extrait.")
        return None
    def extraire_date_naissance(self, texte):
        logger.info(f"Extraction de la date de naissance à partir du texte : {texte}")
        entities = self.nlp(texte)
        for ent in entities:
            if ent['entity_group'] == "DATE":
                date_str = ent['word']
                date_obj = dateparser.parse(date_str)
                if date_obj:
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                    logger.info(f"Date de naissance extraite : {formatted_date}")
                    return formatted_date
                else:
                    logger.warning(f"Date non valide extraites : {date_str}")
                    return date_str
        logger.warning("Aucune date de naissance n'a été extraite.")
        return None
    def extraire_adresse(self, texte):
        logger.info(f"Extraction de l'adresse à partir du texte : {texte}")
        # Extraction du numéro de rue
        numero_rue = re.search(r'\b\d+\b', texte)
        adr = f"{numero_rue.group()} " if numero_rue else ""
        adr=''
        # Extraction des entités pertinentes
        entities = self.nlp(texte)
        for ent in entities:
            if ent['entity_group'] in {"LOC", "PER"}:
                adr += ent['word'] + ' '
        adr = adr.strip()
        if adr:
            logger.info(f"Adresse extraite : {adr}")
        else:
            logger.warning("Aucune adresse n'a été extraite.")
        return adr
    def extraire_numero_telephone(self, texte):
        logger.info(f"Extraction du numéro de téléphone à partir du texte : {texte}")
        # Normalisation du texte en supprimant les espaces, tirets, et points
        phone_number = texte.replace(" ", "").replace("-", "").replace(".", "")
        # Premier regex : validation des numéros compactés
        phone_regex = r"^(\+?\d{1,3})?(\d{9,10})$"
        numero_telephone = re.search(phone_regex, phone_number)
        if numero_telephone:
            logger.info(f"Numéro de téléphone extrait : {numero_telephone.group()}")
            return numero_telephone.group()
        # Deuxième regex : validation des formats avec séparateurs (espaces, tirets)
        numero_telephone = re.search(r"(\+?\d{1,3}[\s-]?)?(\(?\d{1,4}\)?[\s-]?)?(\d{2}[\s-]?){4}\d{2}", phone_number)
        if numero_telephone:
            logger.info(f"Numéro de téléphone extrait avec séparateurs : {numero_telephone.group()}")
            return numero_telephone.group()
        logger.warning("Aucun numéro de téléphone valide n'a été extrait.")
        return None
    def extraire_code_postal(self, texte):
        logger.info(f"Extraction du code postal à partir du texte : {texte}")
        code_postal = re.search(r"\b\d{5}\b", texte)
        if code_postal:
            logger.info(f"Code postal extrait : {code_postal.group()}")
            return code_postal.group()
        else:
            logger.warning("Aucun code postal valide n'a été extrait.")
        return None
    def extraire_adresse_mail(self, texte):
        logger.info(f"Extraction de l'adresse email à partir du texte : {texte}")
        texte = re.sub(r'\s*arobase\s*', '@', texte, flags=re.IGNORECASE)
        adresse_mail = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", texte)
        if adresse_mail:
            logger.info(f"Adresse email extraite : {adresse_mail[0].strip()}")
            return adresse_mail[0].strip()
        else:
            logger.warning("Aucune adresse email valide n'a été extraite.")
        return None

class CreneauExtractor:
    def __init__(self):
        # Initialisation du modèle NLP
        logger.info("Initialisation du modèle NER...")
        self.tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
        self.model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
        self.nlp = pipeline('ner', model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        logger.info("Modèle NER initialisé avec succès.")
        # Dictionnaire de correspondance des nombres en français
        self.french_number_mapping = { "premier": "1", "un": "1", "deux": "2", "trois": "3", "quatre": "4", "cinq": "5",
            "six": "6", "sept": "7", "huit": "8", "neuf": "9", "dix": "10", "onze": "11", "douze": "12", "treize": "13", "quatorze": "14",
            "quinze": "15", "seize": "16", "dix-sept": "17", "dix-huit": "18", "dix-neuf": "19", "vingt": "20", "vingt-et-un": "21",
            "vingt-deux": "22", "vingt-trois": "23", "trente": "30", "trente-et-un": "31", 'minuit': '00h', 'midi': '12h',
            'matin': '9h', 'après-midi': '14h', 'soir': '18h', 'soirée': '18h','deux heures': '2h', 'trois heures': '3h', 'quatre heures': '4h',
            'cinq heures': '5h', 'six heures': '6h', 'sept heures': '7h', 'huit heures': '8h',
            'neuf heures': '9h', 'dix heures': '10h', 'onze heures': '11h', 'douze heures': '12h',
            'treize heures': '13h', 'quatorze heures': '14h', 'quinze heures': '15h',
            'seize heures': '16h', 'dix-sept heures': '17h', 'dix-huit heures': '18h',
            'dix-neuf heures': '19h', 'vingt heures': '20h', 'vingt et une heures': '21h',
            'vingt-deux heures': '22h', 'vingt-trois heures': '23h',}
        
        # Correspondance des jours de la semaine
        self.weekdays_mapping = {"lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, "vendredi": 4, "samedi": 5, "dimanche": 6}

        # Correspondance des dates relatives
        self.relative_dates = {"aujourd'hui": 0,"après-demain": 2,"après demain": 2 ,"apres demain": 2,"demain": 1,  "dans deux jours": 2, "dans trois jours": 3,
            "dans une semaine": 7, "dans deux semaines": 14, "dans trois semaines": 21, "dans un mois": 30, "dans deux mois": 60, "dans trois mois": 90,
            "dans six mois": 180, "dans un an": 365, "dans deux ans": 730, "dans trois ans": 1095,'semaine prochaine':7,'fin du mois' :'30',"mois prochain": 30,
           }
    
    def get_next_weekday(self, target_weekday):
        """Retourne la date du prochain jour donné (0 = Lundi, 6 = Dimanche)."""
        today = datetime.now()
        days_ahead = (target_weekday - today.weekday()) % 7
        return today + timedelta(days=days_ahead if days_ahead else 7)
    
    def convert_french_numbers_to_digits(self, text):
        """Remplace les nombres français par leurs équivalents numériques et normalise les heures."""
        self.logger.debug(f'Conversion des nombres français dans le texte: "{text}"')
        pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in self.french_number_mapping.keys()) + r')\b', re.IGNORECASE)
        text = pattern.sub(lambda match: self.french_number_mapping[match.group(0).lower()], text)
        text = re.sub(r'(\d{1,2})h(?!\d)', r'\1h00', text)  # Normalisation des heures
        return text
    
    def get_end_of_current_month(self):
        """Retourne la date du dernier jour du mois courant."""
        today = datetime.now()
        # Obtenez le dernier jour du mois actuel
        last_day = calendar.monthrange(today.year, today.month)[1]
        end_of_month = today.replace(day=last_day)
        return end_of_month

    def get_first_day_of_next_month(self):
        """Retourne la date du premier jour du mois suivant."""
        today = datetime.now()
        # Ajoutez un mois et réinitialisez le jour à 1
        next_month = today.replace(month=(today.month % 12) + 1, day=1)
        return next_month

    def update_choix_patient(self, choix_patient):
        now = datetime.now()
    
        # Gestion spécifique du "1er du mois prochain"
        if "1er du mois prochain" in choix_patient:
            target_date = self.get_first_day_of_next_month()
            formatted_date = format_date(target_date, format="d MMMM yyyy", locale="fr")
            choix_patient = choix_patient.replace("1er du mois prochain", f"le {formatted_date}")

        if "1er du mois" in choix_patient:
            target_date = self.get_first_day_of_next_month()
            formatted_date = format_date(target_date, format="d MMMM yyyy", locale="fr")
            choix_patient = choix_patient.replace("1er du mois", f"le {formatted_date}")
        
        # Gestion spécifique du "fin du mois"
        if "fin du mois" in choix_patient:
            target_date = self.get_end_of_current_month()
            formatted_date = format_date(target_date, format="d MMMM yyyy", locale="fr")
            choix_patient = choix_patient.replace("fin du mois", f"le {formatted_date}")

        for key, days in self.relative_dates.items():
            if key in choix_patient:
                target_date = datetime.now() + timedelta(days=int(days))
                formatted_date = format_date(target_date, format="d MMMM yyyy", locale="fr")
                choix_patient = choix_patient.replace(key, f"le {formatted_date}")
        
        # Gestion des jours de la semaine
        for jour, index in self.weekdays_mapping.items():
            if f"{jour} prochain" in choix_patient:
                target_date = self.get_next_weekday(index)
                formatted_date = format_date(target_date, format="d MMMM yyyy", locale="fr")
                choix_patient = choix_patient.replace(f"{jour} prochain", f"le {formatted_date}")
        return choix_patient


    def get_creneau(self, choix_patient):
        """Analyse le texte pour extraire une date et une heure."""
        self.logger.info(f"Traitement de l'entrée: {choix_patient}")
        choix_patient = choix_patient.lower()
        choix_patient=self.convert_french_numbers_to_digits(choix_patient)
        choix_patient = re.sub(r'[^\w\s]', ' ', choix_patient)
        choix_patient = re.sub(r'\s+', ' ', choix_patient)
        choix_patient =self.update_choix_patient(choix_patient)
        entities = self.nlp(choix_patient)
        creneau_parts = [str(ent['word']) for ent in entities if ent['entity_group'] in ("DATE", "TIME")]


        if not creneau_parts:
            self.logger.warning('Aucune entité DATE ou TIME détectée.')
            return None
        
        # Combinaison des informations extraites
        creneau_choisi = ' '.join(creneau_parts)
        if 'prochain' in creneau_choisi or 'prochaine' in creneau_choisi:
            creneau_choisi = creneau_choisi.replace('prochain', '').replace('prochaine', '')

        date_obj = dateparser.parse(creneau_choisi, languages=['fr'])
        if date_obj:
            now = datetime.now()
            if date_obj.year == now.year and date_obj < now:
                date_obj = date_obj.replace(year=now.year + 1)
                if date_obj.hour == 0 and date_obj.minute == 0:
                    date_obj = date_obj.replace(hour=9, minute=0)
            return date_obj.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            self.logger.warning(f"Échec de l'analyse de la date: {creneau_choisi}")
            return None


extractor = InformationExtractor()
extractor1=CreneauExtractor()

handlers: Dict[str, callable] = {
    "extraire_nom": extractor.extraire_nom,
    "extraire_prenom": extractor.extraire_prenom,
    "extraire_date_naissance": extractor.extraire_date_naissance,
    "extraire_adresse": extractor.extraire_adresse,
    "extraire_adresse_mail": extractor.extraire_adresse_mail,
    "extraire_code_postal": extractor.extraire_code_postal,
    "extraire_numero_telephone": extractor.extraire_numero_telephone,
    "extraire_creneau": extractor1.get_creneau,
}

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Gère la requête en fonction de l'action demandée"""
    logger.info("Début du traitement de la requête HTTP")

    try:
        req_body = req.get_json()
        logger.info("Corps de la requête JSON récupéré avec succès")
    except ValueError as e:
        logger.error(f"Erreur lors du traitement de la requête : {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON format"}),
            mimetype="application/json",
            status_code=400
        )

    # Extraction et validation des paramètres
    action = req_body.get("action", "").strip()
    texte = req_body.get("texte", "").strip()

    if not action or not texte:
        return func.HttpResponse(
            json.dumps({"error": "Paramètres 'action' et 'texte' requis"}),
            mimetype="application/json",
            status_code=400
        )

    logger.info(f"Action reçue : {action}")
    logger.info(f"Texte reçu : {texte[:50]}...")  # Limite pour éviter d'exposer des données sensibles

    # Vérification si l'action est valide
    handler = handlers.get(action)
    if not handler:
        logger.error(f"Action inconnue : {action}")
        return func.HttpResponse(
            json.dumps({"error": "Action inconnue"}),
            mimetype="application/json",
            status_code=400
        )

    logger.info(f"Exécution de l'action : {action}")

    # Exécuter la fonction correspondante et retourner le résultat
    try:
        result = handler(texte)
        return func.HttpResponse(
            json.dumps({"response": result}),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de l'action '{action}': {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Erreur lors de l'exécution: {str(e)}"}),
            mimetype="application/json",
            status_code=500
        )


