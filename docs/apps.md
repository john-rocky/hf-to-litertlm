# Using a bundle in an app

LiteRT-LM is Google's recommended replacement for the MediaPipe LLM Inference API (`com.google.mediapipe:tasks-genai`), which is in maintenance-only mode.

## Android

In an Android app, a published bundle is one Gradle line and a few lines of Kotlin through
[hfmodels-android](https://github.com/john-rocky/hfmodels-android)
(`implementation("io.github.john-rocky.hfmodels:hfmodels-litertlm:0.1.1")`), pinned to the file
and backend of its device record:

```kotlin
val models = HfModels(applicationContext)
val chat = models.fromPretrained(
    ModelRef("litert-community/LFM2.5-1.2B-Instruct", revision = "f45d8d8abe93bff4026efee20fa483150ce8e687", variant = "int4_gpu"),
    Tasks.Chat, LoadOptions(backendPolicy = BackendPolicy.RequireProfile("gpu"))) { event -> Log.i("hfmodels", event.toString()) }
val session = chat.createConversation(ConversationConfig(systemInstruction = Contents.of("You are a helpful assistant.")))
session.stream(Contents.of("What is 17 + 25? Answer briefly.")).collect { message -> append(message.text) }
withContext(NonCancellable) { models.closeAndJoin() }
```

**Run a published model on Android:** start with
[LFM2.5-1.2B-Instruct](https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct) and the
[hfmodels 0.1.1 install and first-reply guide](https://github.com/john-rocky/hfmodels-android#add-it).
The guide fixes the model revision and GPU profile, includes the imports and cleanup,
and shows the expected **42** reply and where to report a failure. Its tested target is
a **Pixel 8a with 8 GB RAM, Android 16**; installation needs **API 31+, compileSdk 36**
and a 736 MB download plus cache space. [Other SDK entry points](../readers/README.md#sdk-entry-points)
are listed separately with their verification scope.

## iPhone

In an iPhone app, a bundle loads through [swift-litert-lm](https://github.com/john-rocky/swift-litert-lm)
with `LiteRTChat(huggingFaceRepo:fileName:)`; the verified path for a fine-tune is its
[recipe](https://github.com/john-rocky/swift-litert-lm/blob/main/docs/recipe-hf-finetune-to-iphone.md).

**iPhone.** A converted bundle loads in an existing iOS app through swift-litert-lm: [recipe](https://github.com/john-rocky/swift-litert-lm/blob/main/docs/recipe-hf-finetune-to-iphone.md) (one SwiftPM dependency, `LiteRTChat(huggingFaceRepo:fileName:)` or `LiteRTChat(modelFileURL:)`, stop and release, a verify command with its expected output).
