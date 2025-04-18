import dotenv from 'dotenv'
import { ChatGoogleGenerativeAI } from '@langchain/google-genai'
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai'
import { QdrantVectorStore } from '@langchain/qdrant'

import 'cheerio'
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import ora from 'ora'
import readline from 'node:readline/promises'
import { stdin as input, stdout as output } from 'node:process'

import { pull } from 'langchain/hub'
import { Annotation } from '@langchain/langgraph'
import { StateGraph } from '@langchain/langgraph'

dotenv.config()

async function main(web_url, question) {
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
   const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
         url: process.env.QDRANT_URL,
         collectionName: 'langchainjs-testing',
      }
   )

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

   const spinnerIndex = ora('Indexing Process...\n').start()

   try {
      //? Document Loading using CHEERIO
      const pTagSelector = 'p'
      const cheerioLoader = new CheerioWebBaseLoader(web_url, {
         selector: pTagSelector,
      })

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
      const spinnerDocs = ora(
         'Adding documents (Chunks) to vector store...'
      ).start()
      try {
         await vectorStore.addDocuments(allSplits)
         spinnerDocs.succeed('Documents ( Chunks ) added to vector store.')
      } catch (error) {
         console.log('Error adding documents to vector store:', error)
         spinnerDocs.fail('Failed to add documents to vector store.')
      }
      spinnerIndex.succeed('Indexing Process Done.')
   } catch (error) {
      console.log('Error indexing documents:', error)
      spinnerIndex.fail('Failed to index documents.')
   }

   //* INDEXING DONE

   //* RETRIEVING STARTS

   //? Not working bcz of below line - actually it looks like just typescript now, its same if i remove CHATPROMPTTEMPLATE FROM HERe
   // const promptTemplate = (await pull) < ChatPromptTemplate > 'rlm/rag-prompt'

   // // Example:
   // const example_prompt = await promptTemplate.invoke({
   //    context: '(context goes here)',
   //    question: '(question goes here)',
   // })
   // const example_messages = example_prompt.messages

   // console.assert(example_messages.length === 1)
   // example_messages[0].content

   //? CHATPROMPT TEMPLATE INSTANTIATION -working below one, just commented bcz it was for checkign with example its working
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

   const InputStateAnnotation = Annotation.Root({
      question: Annotation,
   })

   const StateAnnotation = Annotation.Root({
      question: Annotation,
      context: Annotation,
      answer: Annotation,
   })

   //? Nodes - Actual Steps - import { concat } from "@langchain/core/utils/stream";

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

   const graph = new StateGraph(StateAnnotation)
      .addNode('retrieve', retrieve)
      .addNode('generate', generate)
      .addEdge('__start__', 'retrieve')
      .addEdge('retrieve', 'generate')
      .addEdge('generate', '__end__')
      .compile()

   //? USING GRAPH TO TRY OUT OUR RAG

   // let inputs = {
   //    question: 'Tell me how to install tailwind css for my react project  ?',
   // }
   let inputs = {
      question: question,
   }

   //? Simple working
   const spinnerStream = ora('Calling graph.stream()...').start()
   try {
      const result = await graph.invoke(inputs)
      // console.log(result.context.slice(0, 2))
      console.log('\n')
      console.log(`Answer: ${result['answer']}`)
      spinnerStream.succeed('Response Retrieved.')
   } catch (error) {
      console.log('Error calling graph.stream():', error)
      spinnerStream.fail('Failed to Retrieve.')
   }

   //? Calling stream steps
   // const spinnerStream = ora('Calling graph.stream()...').start()
   // try {
   //    console.log(inputs)
   //    console.log('\n====\n')
   //    for await (const chunk of await graph.stream(inputs, {
   //       streamMode: 'updates',
   //    })) {
   //       console.log(chunk)
   //       console.log('\n====\n')
   //    }
   //    spinnerStream.succeed('Called graph.stream().')
   // } catch (error) {
   //    console.log('Error calling graph.stream():', error)
   //    spinner.fail('Failed to call graph.stream().')
   // }
}

const rl = readline.createInterface({ input, output })
const web_url = await rl.question('\nEnter the URL of the website: ')
const question = await rl.question('\nEnter your question: ')
console.log('\n')
rl.close()
main(web_url, question)
