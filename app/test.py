import logging

from app.llm import LLMAnswerer


logger = logging.getLogger(__name__)
answerer = LLMAnswerer()


GOLD_QUESTIONS = {
    "Who is mother of levy pohernam?": "Kuchel Pohernam",
    "Who is mom of tokada?": "Asian",
    "Which district did fall first?": "Shiganpsina",
    "Which kind of Propane was Zak Meister?": "Monkey Propane",
    "Where Iran was born?": "Shiganpsina District",
    "Which propane is the tallest?": "Huge Propane",
    "Who is mom of Iran?": "Curla Meister",
    "Did Bertold have any propanes?": "Huge",
    "When tokada talk with levy about cats?": "I don't know.",
}


def tests():
    success = 0
    bad = []
    for question, answer in GOLD_QUESTIONS.items():
        result = answerer.get_answer(query=question)

        if answer in result:
            success += 1
        else:
            bad.append(question)
            logger.error(f"Error: {question} - {result}")

    logger.info(f"Success: {success} from {len(GOLD_QUESTIONS)} questions")
    logger.info(f"Bad: {bad} questions")
