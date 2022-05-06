import logging

logger = logging.getLogger(__name__)


def makeSquad(_id, new_context_list, question, answer):
    passage = "".join(new_context_list)
    if passage == "":
        passage = "无文本"
        logger.info("ID: %s has no passage ...", _id)
    data = {'title': '',
            'paragraphs': [{
                'qas': [{
                    'question': question,
                    'id': _id,
                    'answers': [{
                        'text': answer,
                        'answer_start': -1}],
                    'is_impossible': answer in ['yes', 'no']}],
                'context': passage}]
            }
    return data
