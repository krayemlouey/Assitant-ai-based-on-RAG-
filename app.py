from qa import ask_question

print("RAG Assistant pret (LLaMA + FAISS)")

while True:
    query = input("\nTa question : ")

    if query.lower() in ["exit", "quit"]:
        break

    answer = ask_question(query)

    print("\nReponse :")
    print(answer)
