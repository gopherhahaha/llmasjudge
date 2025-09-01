import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 1. セットアップ ---
# .envファイルから環境変数を読み込む
load_dotenv()

# 評価用LLMの定義
judge_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# --- 2. RAGシステムによる回答生成 ---
def get_rag_answers(query, n_answers=5):
    """
    RAGシステムを実行して、1つの質問に対して複数の回答を生成する関数
    """
    print("RAGシステムを準備しています...")
    # dataフォルダの存在チェックとPDFの読み込み
    folder_name = "data"
    if not os.path.exists(folder_name):
        print(f"エラー: '{folder_name}' ディレクトリが見つかりません。PDFファイルを追加してください。")
        return []

    files = os.listdir(folder_name)
    docs = []
    pdf_files_found = any(file.endswith(".pdf") for file in files)

    if not pdf_files_found:
        print(f"エラー: '{folder_name}' ディレクトリにPDFファイルが見つかりません。")
        return []

    for file in files:
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_name, file))
            docs.extend(loader.load())

    # テキストの分割
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30, separator="\n")
    splitted_pages = text_splitter.split_documents(docs)

    # ベクトルストアの準備
    embeddings = OpenAIEmbeddings()
    db_directory = ".healthX_db"
    if os.path.isdir(db_directory):
        db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(splitted_pages, embedding=embeddings, persist_directory=db_directory)

    # RAGチェーンの構築
    retriever = db.as_retriever()
    rag_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7) # 多様な回答を得るため温度を少し上げる
    chain = RetrievalQA.from_chain_type(
        llm=rag_llm,
        chain_type="stuff",
        retriever=retriever
    )

    # 回答をN個生成
    print(f"'{query}' に対する回答を{n_answers}個生成します...")
    answers = [chain.invoke(query)['result'] for _ in range(n_answers)]
    print("回答の生成が完了しました。")
    return answers

# --- 3. スコアによる評価 ---
def evaluate_by_score(query, answer):
    """
    単一の回答をスコアで評価する関数
    """
    prompt_template = """
    あなたは、提供されたコンテキストに基づいて回答を評価する、公平なAIアシスタントです。
    以下の質問、回答、および評価基準に基づいて、回答の品質を評価してください。

    # 質問
    {query}

    # 回答
    {answer}

    # 評価基準
    1. 関連性: 回答は質問に直接関連していますか？
    2. 正確性: 回答に含まれる情報は、提供されたコンテキストに照らして正確ですか？
    3. 網羅性: 回答は質問のすべての側面をカバーしていますか？
    4. 明瞭性: 回答は明確で理解しやすいですか？
    5. 簡潔性: 回答は冗長でなく、要点をまとめていますか？
    6. 有用性: 回答はユーザーにとって役立つものですか？

    # 指示
    各評価基準について、1から5の5段階でスコアを付けてください（1: 不十分, 5: 非常に良い）。
    また、それぞれのスコアの理由と、総合的な評価（1-5）とコメントをJSON形式で出力してください。
    出力はJSONオブジェクトのみとし、前後に他のテキストを含めないでください。
    {{
      "scores": {{
        "relevance": <score>,
        "accuracy": <score>,
        "completeness": <score>,
        "clarity": <score>,
        "conciseness": <score>,
        "usefulness": <score>
      }},
      "justification": "<各スコアの理由>",
      "overall_score": <score>,
      "overall_comment": "<総合的なコメント>"
    }}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["query", "answer"])
    chain = prompt | judge_llm
    response = chain.invoke({"query": query, "answer": answer})
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response from LLM.", "response": response.content}


# --- 4. 回答比較による評価 ---
def evaluate_by_comparison(query, answers):
    """
    複数の回答を比較してランキング付けする関数
    """
    formatted_answers = "\n\n".join([f"--- 回答 {i+1} ---\n{ans}" for i, ans in enumerate(answers)])
    
    prompt_template = """
    あなたは、提供されたコンテキストに基づいて複数の回答を比較評価する、公平なAIアシスタントです。
    以下の質問と複数の回答候補を読み、評価基準に基づいて最も優れた回答を1つ選んでください。

    # 質問
    {query}

    # 回答候補
    {answers}

    # 評価基準
    1. 関連性: 回答は質問に直接関連していますか？
    2. 正確性: 回答に含まれる情報は、提供されたコンテキストに照らして正確ですか？
    3. 網羅性: 回答は質問のすべての側面をカバーしていますか？
    4. 明瞭性: 回答は明確で理解しやすいですか？
    5. 簡潔性: 回答は冗長でなく、要点をまとめていますか？
    6. 有用性: 回答はユーザーにとって役立つものですか？

    # 指示
    すべての回答候補を評価基準に照らして比較し、最も優れているものから順にランキングを付けてください。
    それぞれの回答について、なぜその順位になったのか、具体的な理由を説明してください。
    最終的な結果をJSON形式で出力してください。
    出力はJSONオブジェクトのみとし、前後に他のテキストを含めないでください。
    {{
      "ranking": [
        {{
          "rank": 1,
          "answer_index": <index>,
          "justification": "<なぜこの回答が1位なのかの説明>"
        }},
        {{
          "rank": 2,
          "answer_index": <index>,
          "justification": "<なぜこの回答が2位なのかの説明>"
        }}
      ],
      "best_answer_index": <最も優れた回答のインデックス>
    }}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["query", "answers"])
    chain = prompt | judge_llm
    response = chain.invoke({"query": query, "answers": formatted_answers})
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response from LLM.", "response": response.content}

# --- 5. メイン処理 ---
if __name__ == "__main__":
    USER_QUERY = "HealthXの無料プランについて教えてください。"
    NUM_ANSWERS = 5

    # RAGシステムから回答を取得
    rag_answers = get_rag_answers(USER_QUERY, NUM_ANSWERS)

    if rag_answers:
        print("\n" + "="*20 + " スコアによる評価 " + "="*20)
        for i, ans in enumerate(rag_answers):
            print(f"\n--- 回答 {i+1} の評価 ---")
            print(f"回答内容: {ans}")
            score_result = evaluate_by_score(USER_QUERY, ans)
            print("評価結果:")
            print(json.dumps(score_result, indent=2, ensure_ascii=False))

        print("\n" + "="*20 + " 回答比較による評価 " + "="*20)
        comparison_result = evaluate_by_comparison(USER_QUERY, rag_answers)
        print("比較評価結果:")
        print(json.dumps(comparison_result, indent=2, ensure_ascii=False))
        
        if "best_answer_index" in comparison_result:
            best_index = comparison_result["best_answer_index"]
            if 0 <= best_index < len(rag_answers):
                 print("\n--- 最も優れた回答 ---")
                 print(rag_answers[best_index])
