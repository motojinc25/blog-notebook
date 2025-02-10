---
title: "Semantic Kernel入門 - (2)テンプレート"
emoji: "🧠"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ['semantickernel','python','openai']
published: true
---

# 概要

前回は Semantic Kernel（セマンティック カーネル）の Chat Completion を使って、Hello World という簡単なプロンプトを渡し、その応答を取得して出力してみました。
Chat Completion を利用する場合は会話履歴の管理が必要ですが、最小限の設定のみを行いました。
今回はもう少し会話履歴を活用できるように、テンプレートを用いて動的にプロンプトを生成する方法を紹介します。

# テンプレートのフォーマット

Semantic Kernelはテンプレートの中身を記載するルールとしていくつかサポートしています。
テンプレートを記載するルールはSemantic Kernelがサポートする言語によってサポートできるフォーマットも変わりますので、ご注意ください。
こちらでは、Semantic Kernel Prompt Template Syntaxについて調べてみます。

Semantic Kernel では、テンプレートを記述するための構文としていくつかのルールをサポートしています。
ただし、Semantic Kernel がサポートするプログラミング言語によって、利用できるテンプレート構文やフォーマットが異なる場合があります。
ここでは、[Semantic Kernel Prompt Template Syntax](https://learn.microsoft.com/ja-jp/semantic-kernel/concepts/prompts/prompt-template-syntax) について簡単に見ていきます。

![](/images/sk_20250211_prompttemplate/template_format.png)

## 変数

最も基本的なルールとして、テンプレート内で `{{ ... }}` に囲まれた箇所に `$変数名` を記述すると、その変数が該当箇所に展開されます。

```plain
Hello {{ $name }}, welcome to Semantic Kernel!
```

## 関数の呼び出し

外部関数を呼び出すには、`{{ ... }}` の中に関数名を直接記載します。パラメータを含める場合は、関数名の後に変数や文字列を指定します。

```plain
The weather today in {{ $city }} is {{ weather.getForecast $city }}.
The weather today in Tokyo is {{weather.getForecast "Tokyo"}}.
```

# Semantic Kernel で Prompt Template を使う

## Notebook の参考

GitHub には実際に動作するサンプルコードを Notebook として用意しています。結果をすぐに確認したい方はぜひこちらもご参照ください。

https://github.com/motojinc25/blog-notebook/blob/main/notebooks/semantic-kernel/02_prompt_template.ipynb

## Prompt Template を使って会話する

### 1. Kernel（カーネル）のインスタンス作成

まずは、Semantic Kernel のメインコンポーネントである `Kernel` のインスタンスを作成します。

```python
from semantic_kernel import Kernel
kernel = Kernel()
```

### 2. AIサービス（Azure OpenAI Service）の設定

Semantic Kernel では、さまざまな AI サービスを登録して使用できます。
ここでは Azure OpenAI Service の GPT-4o モデルを利用し、[Chat Completion API](https://learn.microsoft.com/ja-jp/azure/ai-services/openai/how-to/chatgpt)を使う例を示します。

> **補足**: サービス ID（`service_id`）は必須ではありませんが、複数のサービスを使い分ける場合などに指定しておくと管理が楽になります。

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

会話の履歴を扱うため、チャット履歴オブジェクトを作成します。

```python
from semantic_kernel.contents import ChatHistory
chat_history = ChatHistory()
```

### 4. プロンプトテンプレートを作成して登録する

今回のチャットでは、過去の会話履歴とユーザー入力を組み合わせたプロンプトを使いたいので、テンプレートを用意します。
`"semantic-kernel"` 形式のテンプレートを使用し、過去の会話履歴とユーザーの入力が埋め込まれるように設定します。


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
    execution_settings=AzureChatPromptExecutionSettings(),
)

# Kernel（カーネル）にプロンプトテンプレートを関数として登録
chat_function = kernel.add_function(
    function_name="chat",
    plugin_name="ChatBot",
    prompt_template_config=prompt_template_config,
)
```

### 5. 会話を開始し、AI の応答を受け取る

ユーザー入力を送信し、Chat Completion API からの応答を取得してみます。
テンプレートで利用する変数は `KernelArguments` オブジェクトとして渡します。

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

続けて、別の入力を行い、会話履歴がどのように反映されるかを確認します。

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

Semantic Kernel でプロンプトテンプレートを活用することで、プロンプトの再利用性や可読性を高められます。
今後もこのシリーズで、Semantic Kernel のさまざまな便利機能を順次紹介していきます。
一緒に Semantic Kernel を使いながら、学んでいきましょう！ 🚀
