---
title: "Semantic Kernel入門 - (1)Hello World"
emoji: "🧠"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ['semantickernel','python','openai']
published: true
---

# 概要

Semantic Kernel（セマンティック カーネル） は、AIを活用したアプリケーションを効率よく開発できる軽量なフレームワークです。
LangChainなどと同じAIオーケストレーターのカテゴリに分類され、複数のAIモデルやツールを連携させることで、より高度なタスクを実行できます。

Semantic Kernelは、Microsoft社が[オープンソース](https://github.com/microsoft/semantic-kernel)として提供・管理しており、Java・C#・Pythonで利用可能です。
このシリーズでは、PythonをベースにSemantic Kernelの基本的な使い方から応用的な活用方法までを、デモを交えながら紹介していきます。

# Semantic Kernelとは

一言で表すと、AI Orchestration（オーケストレーション）を実現するフレームワークです。
AIオーケストレーションとは、複数のAIモデルやアルゴリズムを組み合わせて、より高度・複雑なタスクを実行する仕組みのことを指します。

Microsoft社は「Copilot Stack」という概念を提唱しており、Semantic Kernelはこの中の *AI Orchestration* を担う重要な役割を果たします。

![image](/images/sk_20250131_helloworld/copilot-stack.jpg)

Semantic Kernelを活用すると、単一のAIモデルでは実現が難しい複数のタスクを統合したスマートなアプリケーションを構築できます。
このシリーズを通じて、Semantic Kernelの機能を試しながら、さまざまな用途に活用できる可能性を探っていきましょう。

# Hello Worldの出力

まずは、最もシンプルな例として 「Hello World」 を出力するサンプルを実行してみます。
Semantic Kernelを使ってAzure OpenAI Serviceの言語モデルに「Hello World」のプロンプトを渡し、それに対する応答を取得して出力します。
この基本的なプロセスを理解することで、Semantic Kernelの使い方のイメージを掴むことができます。

## Notebook参考

GitHubにはコードのデモを用意しています。結果を確認する場合は、ぜひGitHubのNotebookをご参照ください。

https://github.com/motojinc25/blog-notebook/blob/main/notebooks/semantic-kernel/01_helloworld.ipynb

## 環境を準備しよう

まず、Semantic Kernelと必要なパッケージをインストールしてバージョンを確認ます。

```python
pip install -qU semantic-kernel python-dotenv
from semantic_kernel import __version__
print(__version__)
```

次に、Azure OpenAI Serviceのキー情報を保存している「.env」ファイルを読み込みます。
`.env` ファイルには、次のような設定を記述します。

```bash
SK_AZURE_OPENAI_DEPLOYMENT_NAME=***
SK_AZURE_OPENAI_API_KEY=***
SK_AZURE_OPENAI_ENDPOINT=***
SK_AZURE_OPENAI_API_VERSION=***
```

この情報をコードで読み込みます。

```python
from dotenv import load_dotenv
load_dotenv()
```

## Hello Worldを出力しよう

### 1. Kernel（カーネル）を作成する

Semantic Kernelのメインコンポーネントである Kernel（カーネル）のインスタンスを作成します。
今回は必須ではありませんが、Semantic Kernelの流れに慣れるために設定しておきます。

```python
from semantic_kernel import Kernel
kernel = Kernel()
```

### 2. AIサービス（Azure OpenAI Service）を設定する

Semantic Kernelでは、AIサービスを登録して使用します。
ここでは、Azure OpenAI Serviceの GPT-4oモデルを利用し、[Chat Completion API](https://learn.microsoft.com/ja-jp/azure/ai-services/openai/how-to/chatgpt)を使います。
サービスIDは必須ではありませんが、流れを理解しやすくするために設定しておきます。

```python
import os
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
service_id = None
chat_completion_service = AzureChatCompletion(
    deployment_name=os.getenv("SK_AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("SK_AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("SK_AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("SK_AZURE_OPENAI_API_VERSION"),
    service_id=service_id,
)
kernel.add_service(chat_completion_service)
```

### 3. チャット履歴を作成する

チャットセッション内でメッセージ履歴を管理するために、チャット履歴オブジェクトを作成します。

```python
from semantic_kernel.contents import ChatHistory
chat_history = ChatHistory()
```

### 4. プロンプトの設定を作成する

Chat Completion APIをSemantic Kernelを経由して呼び出す際に、プロンプトの設定オブジェクトを作成します。

```python
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
execution_settings = AzureChatPromptExecutionSettings()
```

### 5. 「Hello World」を送信し、AIの応答を受け取る

ユーザーの入力として「Hello World」を送信し、Chat Completion APIから応答を取得して出力します。

```python
chat_history.add_user_message("Hello World")
response = await chat_completion_service.get_chat_message_content(
    chat_history=chat_history,
    settings=execution_settings,
    kernel=kernel,
)
print(response)
```

# まとめ

Semantic Kernelでは、登録したAIサービスを抽象化して扱うため、異なるLLM（大規模言語モデル）を 同じコードで簡単に切り替えられるというメリットがあります。
これからのシリーズで、Semantic Kernelの便利な機能をどんどん紹介していきます。
一緒に楽しみながら、Semantic Kernelを学んでいきましょう！ 🚀
