import chainlit as cl
from Exercise2 import run_rag


@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "Ask me questions about:\n"
            "- MachineLearning-Lecture01.pdf\n"
            "- donut_paper.pdf\n"
            "- winter-sports.pdf"
        )
    ).send()


@cl.on_message
async def answer_question(message: cl.Message):
    thinking_message = cl.Message(content="Searching the documents...")
    await thinking_message.send()

    try:
        result = await cl.make_async(run_rag)(
            {"question": message.content}
        )

        sources = ", ".join(result["retrieved_sources"])
        pages = ", ".join(
            f"{item['source']} — page {item['page']}"
            for item in result["retrieved_pages"]
        )

        thinking_message.content = (
            f"{result['answer']}\n\n"
            f"**Retrieved sources:** {sources}\n\n"
            f"**Retrieved pages:** {pages}\n\n"
            f"**Response time:** "
            f"{result['total_time_seconds']:.2f} seconds"
        )

        await thinking_message.update()

    except Exception as error:
        thinking_message.content = f"Error: {error}"
        await thinking_message.update()