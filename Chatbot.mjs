import dotenv from 'dotenv'
import { ChatGoogleGenerativeAI } from '@langchain/google-genai'
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai'
import { QdrantVectorStore } from '@langchain/qdrant'

import 'cheerio'
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'

dotenv.config()

//* INSTANTIATING PROCESS START
//? Instantiating GEMINI CHAT
const llm = new ChatGoogleGenerativeAI({
   model: 'gemini-2.0-flash',
   temperature: 0,
   // maxRetries: 2,
   // other params...
})

//? Embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
   // model: 'text-embedding-3-large', // 768 dimensions
   model: 'text-embedding-004',
})

//? Vector Store
const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
   url: process.env.QDRANT_URL,
   collectionName: 'langchainjs-testing',
})
//* INSTANTIATING PROCESS Done

//? ChatBot Calling
// async function botCall() {
//    try {
//       const response = await llm.invoke([
//          {
//             role: 'system',
//             content:
//                'You are a helpful assistant that translates English to French. Translate the user sentence.',
//          },
//          { role: 'user', content: 'Hello' },
//       ])

//       console.log('Response:', response.content)
//    } catch (error) {
//       console.log('Error invoking Gemini:', error)
//    }
// }
// botCall()

//* INDEXING PROCESS START

//? Document Loading using CHEERIO
const pTagSelector = 'p'
const cheerioLoader = new CheerioWebBaseLoader(
   'https://lilianweng.github.io/posts/2023-06-23-agent/',
   {
      selector: pTagSelector,
   }
)

const docs = await cheerioLoader.load()

// console.assert(docs.length === 1)
// console.log(`Total characters: ${docs[0].pageContent.length}`)
// console.log(docs.length)

//? TEXT SPLITTING USING RECCURSIVECHRACTEERTEXTSPLITTER

const splitter = new RecursiveCharacterTextSplitter({
   chunkSize: 1000,
   chunkOverlap: 200,
})

const allSplits = await splitter.splitDocuments(docs)
// console.log(`Split blog post into ${allSplits.length} sub-documents.`)
// console.log(allSplits[5])

//? Storing Chunks in Vector Store
try {
   await vectorStore.addDocuments(allSplits)
   console.log('Documents added to vector store.')
} catch (error) {
   console.log('Error adding documents to vector store:', error)
}
//* INDEXING DONE

//* RETRIEVING STARTS
import { pull } from 'langchain/hub'
import { ChatPromptTemplate } from '@langchain/core/prompts'

// const promptTemplate = (await pull) < ChatPromptTemplate > 'rlm/rag-prompt'

// // Example:
// const example_prompt = await promptTemplate.invoke({
//    context: '(context goes here)',
//    question: '(question goes here)',
// })
// const example_messages = example_prompt.messages

// console.assert(example_messages.length === 1)
// example_messages[0].content

//? CHATPROMPT TEMPLATE INSTANTIATION
// async function getPrompt() {
//    const prompt = await pull('rlm/rag-prompt')
//    if (prompt instanceof ChatPromptTemplate) {
//       const example_prompt = await prompt.invoke({
//          context: '(context goes here)',
//          question: '(question goes here)',
//       })
//       const example_messages = example_prompt.messages
//       console.assert(example_messages.length === 1)
//       console.log(example_messages[0].content)
//    } else {
//       console.error('Pulled prompt is not a ChatPromptTemplate:', prompt)
//    }
// }

// getPrompt()
const promptTemplate = await pull('rlm/rag-prompt')

//
// * USING LANGGRAPH TO WRAP EVERYTHING INSIDE ONE FLOW

//? States for langgraph - contains questions and responses at each step (NODES)
// import { Document } from '@langchain/core/documents'
import { Annotation } from '@langchain/langgraph'

const InputStateAnnotation = Annotation.Root({
   question: Annotation,
})

const StateAnnotation = Annotation.Root({
   question: Annotation,
   context: Annotation,
   answer: Annotation,
})

//? Nodes - Actual Steps - import { concat } from "@langchain/core/utils/stream";
import { concat } from '@langchain/core/utils/stream'

const retrieve = async (state) => {
   const retrievedDocs = await vectorStore.similaritySearch(state.question)
   return { context: retrievedDocs }
}

const generate = async (state) => {
   const docsContent = state.context.map((doc) => doc.pageContent).join('\n')
   const messages = await promptTemplate.invoke({
      question: state.question,
      context: docsContent,
   })
   const response = await llm.invoke(messages)
   return { answer: response.content }
}

//? DEFINING  CONTROL FLOW
import { StateGraph } from '@langchain/langgraph'

const graph = new StateGraph(StateAnnotation)
   .addNode('retrieve', retrieve)
   .addNode('generate', generate)
   .addEdge('__start__', 'retrieve')
   .addEdge('retrieve', 'generate')
   .addEdge('generate', '__end__')
   .compile()

//? USING GRAPH TO TRY OUT OUR RAG
let inputs = { question: 'What is Task Decomposition?' }

//? Simple working
// const result = await graph.invoke(inputs)
// console.log(result.context.slice(0, 2))
// console.log(`\nAnswer: ${result['answer']}`)

//? Calling stream steps
console.log(inputs)
console.log('\n====\n')
for await (const chunk of await graph.stream(inputs, {
   streamMode: 'updates',
})) {
   console.log(chunk)
   console.log('\n====\n')
}
