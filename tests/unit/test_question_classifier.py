"""Question routing."""

from __future__ import annotations

import pytest

from app.ai.question import QuestionClassifier, QuestionKind


@pytest.fixture
def classifier() -> QuestionClassifier:
    return QuestionClassifier()


class TestConversationQuestions:
    @pytest.mark.parametrize(
        "question",
        [
            "what was the 2nd question i asked in this convo?",
            "in your earlier response you shared first excerpt, where did you get?",
            "where is these 1st and second excerpt mentioned?",
            "Can you summarise our conversation",
            "You said the limit was 500, is that right?",
            "What did I ask you before?",
            "repeat that answer please",
        ],
    )
    def test_questions_about_the_exchange_are_routed_to_history(
        self, classifier: QuestionClassifier, question: str
    ) -> None:
        assert (
            classifier.classify(question, has_history=True)
            is QuestionKind.CONVERSATION
        )

    def test_without_history_they_fall_through_to_the_documents(
        self, classifier: QuestionClassifier
    ) -> None:
        """On the first turn there is no transcript to answer from."""
        assert (
            classifier.classify("what did i ask?", has_history=False)
            is QuestionKind.DOCUMENT
        )


class TestAggregateQuestions:
    @pytest.mark.parametrize(
        "question",
        [
            "Can you count there are how many EXCLUSIONS listed?",
            "How many exclusions are there?",
            "List all the conditions",
            "What is the total number of clauses?",
            "Give me a complete list of benefits",
            "enumerate the definitions",
            "What are all the fees?",
        ],
    )
    def test_counting_and_listing_is_recognised(
        self, classifier: QuestionClassifier, question: str
    ) -> None:
        assert classifier.classify(question, has_history=True) is QuestionKind.AGGREGATE


class TestDocumentQuestions:
    @pytest.mark.parametrize(
        "question",
        [
            "What are the Geographical Limits?",
            "What is the mean of Insured Person",
            "Where is the termination clause mentioned?",
            "What did the company say about refunds?",
            "Explain the cashless facility",
            "Is maternity covered?",
        ],
    )
    def test_ordinary_questions_go_to_retrieval(
        self, classifier: QuestionClassifier, question: str
    ) -> None:
        assert classifier.classify(question, has_history=True) is QuestionKind.DOCUMENT

    def test_an_empty_question_is_treated_as_a_document_question(
        self, classifier: QuestionClassifier
    ) -> None:
        assert classifier.classify("   ", has_history=True) is QuestionKind.DOCUMENT

    def test_a_document_question_is_not_mistaken_for_a_meta_question(
        self, classifier: QuestionClassifier
    ) -> None:
        """"Where is X mentioned" asks about the document, not the chat."""
        assert (
            classifier.classify("Where is the geographical limit mentioned?", has_history=True)
            is QuestionKind.DOCUMENT
        )


class TestAdverbsBetweenAuxiliaryAndVerb:
    """English drops adverbs mid-phrase: "did I *just* ask".

    The patterns originally matched "what did I ask?" but not "what did I just
    ask you?", so the same question routed two different ways depending on one
    filler word.
    """

    @pytest.mark.parametrize(
        "question",
        [
            "what did I just ask you?",
            "what did I actually ask you earlier?",
            "did we already ask that?",
            "did I previously ask about the excess?",
            "you just said the excess is 10 percent",
            "you already told me the waiting period",
            "I previously asked about geographical limits",
            "we also asked about day care",
        ],
    )
    def test_an_adverb_does_not_hide_a_conversation_question(
        self, question: str
    ) -> None:
        classifier = QuestionClassifier()

        assert (
            classifier.classify(question, has_history=True)
            is QuestionKind.CONVERSATION
        )

    @pytest.mark.parametrize(
        "question",
        [
            "does the policy ask for a deposit?",
            "who should I ask about claims?",
            "what is the excess?",
            "you must ask the insurer within 30 days",
        ],
    )
    def test_ordinary_questions_are_still_document_questions(
        self, question: str
    ) -> None:
        """The adverb group must not widen the patterns into false positives."""
        classifier = QuestionClassifier()

        assert (
            classifier.classify(question, has_history=True)
            is QuestionKind.DOCUMENT
        )
