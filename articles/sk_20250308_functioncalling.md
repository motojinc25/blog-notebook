---
title: "Semantic Kernel入門 - (3)Function Calling"
emoji: "🧠"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ['semantickernel','python','openai']
published: true
---

# 概要

最近の大規模言語モデル（LLM）では、外部の関数を呼び出せるかどうかを応答として返してくれる「Function Calling」という機能が標準でサポートされるようになりました。
本記事では、Azure OpenAI Service の GPT-4o モデルを例に、Semantic Kernel 上で Function Calling を使ってみる手順やポイントを簡単に解説します。

## Function Calling とは

Function Calling とは、Azure OpenAI Service などにデプロイされた LLM（例：GPT-4o）が、外部関数を呼び出すタイミングを検出し、自動的に教えてくれる仕組みです。
具体的には、LLM に対して 「呼び出せる関数やプラグインの情報」 を渡しておくと、モデルが 「この関数を使えばユーザーの要望に応えられそう」 と判断して、どの関数をどのようなパラメーターで呼ぶかを提案してくれます。

https://platform.openai.com/docs/guides/function-calling

Semantic Kernel では、Function Calling に対応したモデル（GPT-4o など）にプロンプトとプラグイン情報を与えて、「どのプラグインを何回呼び出すのか」 という 実行計画 を返してもらいます。

## Planing とは

Semantic Kernel には、Function Calling を使った実行計画を「Planing」と呼ぶ仕組みがあります。
以前のバージョンでは「Planner」という機能を用いて計画を立てていましたが、現在は Function Calling ベースの実行計画のみをサポート する方針に変わり、Planner は非推奨となっています。
Planing（実行計画）は、Function Calling と Semantic Kernel の組み合わせだけでほぼ完結するため、開発者が特別な実装を用意する必要はありません。これを Auto Function Calling と呼ぶこともあります。
実はこのような Function Calling は、もともと OpenAI から始まり、Gemini や Claude などほかの AI モデルでも利用できるようになってきています。いまでは多くの LLM が共通してサポートしている機能になりつつあるのです。

https://learn.microsoft.com/en-us/semantic-kernel/concepts/planning?pivots=programming-language-python

## Auto Function Calling とは

下図は、Semantic Kernel の Auto Function Calling の仕組みを示したイメージです。

![Image](https://learn.microsoft.com/en-us/semantic-kernel/media/functioncalling.png)

### フローの流れ

1. Semantic Kernel で使用できる関数とパラメーター情報 は、あらかじめ JSON スキーマ としてシリアル化されます。
2. シリアル化された関数情報とチャット履歴を、LLM への入力として送信します。
3. LLM は入力を処理し、応答を生成しますが、その応答は「Chat Message」か「Function Calling」のどちらかの形式になります。
4. 応答が Chat Message だった場合は、そのままユーザーに返して画面表示などに利用します。一方で Function Calling だった場合は、返ってきた「関数名」と「パラメーター」を取り出します。
5. 取り出した 関数名とパラメーター を使って、Semantic Kernel 内で対応する関数を呼び出します。
6. 関数の結果は再びチャット履歴として LLM に送られ、2〜6 の流れを LLM から「終了」の合図があるまで繰り返す 仕組みになっています。

# Semantic Kernel で Function Calling を使う

それでは、実際に Semantic Kernel を使って Function Calling を試す手順をご紹介します。

## Notebook の参考

GitHub には、実際に動くサンプルコードを載せた Notebook があります。
手っ取り早く結果を見たい場合は、こちらを参照するとイメージしやすいと思います。

https://github.com/motojinc25/blog-notebook/blob/main/notebooks/semantic-kernel/03_function_calling.ipynb

## Function Calling を使って会話する

### 1. Kernel（カーネル）のインスタンス作成

まずは、Semantic Kernel の中心となるクラス `Kernel` のインスタンスを作成します。

```python
from semantic_kernel import Kernel
kernel = Kernel()
```

### 2. AIサービス（Azure OpenAI Service）の設定

Semantic Kernel では、さまざまな AI サービスを登録して使用できます。
ここでは Azure OpenAI Service の GPT-4o モデルを利用し、[Chat Completion API](https://learn.microsoft.com/ja-jp/azure/ai-services/openai/how-to/chatgpt)を使う例を示します。

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

会話の履歴を扱うため、ChatHistory オブジェクトを作成します。

```python
from semantic_kernel.contents import ChatHistory
chat_history = ChatHistory()
```

### 4. コアー関数を登録する

Function Calling の候補として使える コアー関数 を登録します。
今回は、Semantic Kernel が標準でサポートしている 2 つの関数（MathPlugin / TimePlugin）を例として使います。

```python
from semantic_kernel.core_plugins import MathPlugin, TimePlugin
kernel.add_plugin(MathPlugin(), plugin_name="math")
kernel.add_plugin(TimePlugin(), plugin_name="time")
```

### 5. プロンプトテンプレートを作成して登録する

今回のチャットでは、過去の会話履歴（chat_history）とユーザーの入力（request）を組み合わせるためのプロンプトを準備します。
"semantic-kernel" 形式のテンプレートで、両方の情報が埋め込まれるように設定します。

```python
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
template = """
Previous information from chat:
{{$chat_history}}

User: {{$request}}
Assistant:
"""
prompt_template_config = PromptTemplateConfig(
    template=template,
    name="chat",
    description="Chat with the assistant",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="chat_history", description="The conversation history", is_required=False, default=""),
        InputVariable(name="request", description="The user's request", is_required=True),
    ],
    execution_settings=AzureChatPromptExecutionSettings(function_choice_behavior="auto"),
)
```

Kernel（カーネル）にプロンプトテンプレートを使って関数として登録します。

```python
chat_function = kernel.add_function(
    function_name="chat",
    plugin_name="ChatBot",
    prompt_template_config=prompt_template_config,
)
```

### 6. 会話を開始し、AI の応答を受け取る

ユーザー入力を送信し、Chat Completion API からの応答を取得してみます。
テンプレートで利用する変数は `KernelArguments` オブジェクトとして渡します。

ローカルの時間を出力してみます。

```python
!echo %time%
```

Kernel（カーネル）に登録したコアー関数をFunction Callingによって呼び出されるように現在の時間を聞いてみます。

```python
from semantic_kernel.functions import KernelArguments
# 初回のユーザー入力
user_input = "Hello, how are you? I'm Jin"
answer = await kernel.invoke(
    function=chat_function,
    arguments=KernelArguments(
        request=user_input,
        chat_history=chat_history,
    ),
)
print("User:", user_input)
chat_history.add_user_message(user_input)
print("Assistant:", answer)
chat_history.add_assistant_message(str(answer))
```

履歴の確認と、数値計算のコアー関数がFunction Callingで呼び出されるように聞いてみます。

```python
user_input = "What is my name?"
answer = await kernel.invoke(
    function=chat_function,
    arguments=KernelArguments(
        request=user_input,
        chat_history=chat_history,
    ),
)
print("User:", user_input)
print("Assistant:", answer)
```

## まとめ

Semantic Kernel の Auto Function Calling を活用すると、LLM モデルが自分で関数を呼ぶタイミングを判断してくれるようになるため、モデル単体ではできないような機能もかんたんに実装できます。
今後もこのシリーズで、Semantic Kernel のさまざまな機能を紹介していきますので、ぜひ一緒に学んでいきましょう！ 🚀
