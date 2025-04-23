import dotenv from 'dotenv'
import { ChatGoogleGenerativeAI } from '@langchain/google-genai'
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai'
import { QdrantVectorStore } from '@langchain/qdrant'

import 'cheerio'
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import ora from 'ora'
import chalk from 'chalk'
import readline from 'node:readline/promises'
import { stdin as input, stdout as output } from 'node:process'

import { pull } from 'langchain/hub'
import { Annotation } from '@langchain/langgraph'
import { StateGraph } from '@langchain/langgraph'
import fs from 'fs'
import { exec } from 'node:child_process'

dotenv.config()

async function main(web_url, question) {
   //GlobalVariables
   let llm
   let vectorStore

   //* INSTANTIATING PROCESS START
   const spinnerInstantiate = ora('Instantiating...\n').start()
   try {
      //? Instantiating GEMINI CHAT
      llm = new ChatGoogleGenerativeAI({
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
      vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
         url: process.env.QDRANT_URL,
         collectionName: 'langchainjs-testing',
      })
      spinnerInstantiate.succeed('Instantiating Done.\n')
   } catch (error) {
      console.log('Error during instantiation:', error)
      spinnerInstantiate.fail('Failed to instantiate.')
   }
   //* INSTANTIATING PROCESS Done

   //* INDEXING PROCESS START
   const spinnerIndex = ora('Indexing Process...\n').start()
   try {
      //? Document Loading using CHEERIO
      const pTagSelector = 'p'
      const cheerioLoader = new CheerioWebBaseLoader(web_url, {
         selector: pTagSelector,
      })

      const docs = await cheerioLoader.load()

      //? TEXT SPLITTING USING RECCURSIVECHRACTEERTEXTSPLITTER

      const splitter = new RecursiveCharacterTextSplitter({
         chunkSize: 1000,
         chunkOverlap: 200,
      })

      const allSplits = await splitter.splitDocuments(docs)

      //? Storing Chunks in Vector Store
      const spinnerDocs = ora(
         'Adding documents (Chunks) to vector store...'
      ).start()
      try {
         await vectorStore.addDocuments(allSplits)
         spinnerDocs.succeed('Documents ( Chunks ) added to vector store.\n')
      } catch (error) {
         console.log('Error adding documents to vector store:', error)
         spinnerDocs.fail('Failed to add documents to vector store.\n')
      }
      spinnerIndex.succeed('Indexing Process Done.\n')
   } catch (error) {
      console.log('Error indexing documents:', error)
      spinnerIndex.fail('Failed to index documents.\n')
   }
   //* INDEXING DONE

   //* RETRIEVING STARTS
   const promptTemplate = await pull('rlm/rag-prompt')

   const StateAnnotation = Annotation.Root({
      question: Annotation,
      context: Annotation,
      answer: Annotation,
   })

   let docContent
   let result

   //? 1-  PROMPT RUNNING PART

   const queryGeneratorPrompt = `
   You are an Intelligent Assistant who writes a paragraph to answer a question .
   Goal is to generate a paragraph to answer "${question}" in JSON format.
   Output ONLY with a raw JSON Array, no explanation, no code block, no backticks. Example: ["content"]
   `
   const queryResponse = await llm.invoke(queryGeneratorPrompt)
   let cleanedContent = queryResponse.content.trim()
   if (cleanedContent.startsWith('```')) {
      cleanedContent = cleanedContent.replace(/```(json)?/g, '').trim()
   }
   try {
      docContent = JSON.parse(cleanedContent) // parse the LLM response
   } catch (err) {
      console.error('Error parsing LLM generated queries:', err)
      docContent = [question] // fallback: just use the base question
   }
   console.log('\nGenerated Hypothetical Doc:', docContent)

   //? 3-  RETRIEVING FXN

   const retrieve = async (state) => {
      const resultsArray = await vectorStore.similaritySearch(state.question, 2)
      // console.log(
      //    '\nResults array after generating Vector embeddings based on DocContent (Hypothetical doc we created): ',
      //    resultsArray
      // )
      return {
         context: resultsArray,
      }
   }

   //? 4- GENERATE FXN
   const generate = async (state) => {
      const docsContent = state.context.map((doc) => doc.pageContent).join('\n')
      const messages = await promptTemplate.invoke({
         question: state.question,
         context: docsContent,
      })
      const response = await llm.invoke(messages)
      return { answer: response.content }
   }

   //? DEFINING GRAPH NODES AND EDGES
   const graph = new StateGraph(StateAnnotation)
      .addNode('retrieve', retrieve)
      .addNode('generate', generate)
      .addEdge('__start__', 'retrieve')
      .addEdge('retrieve', 'generate')
      .addEdge('generate', '__end__')
      .compile()

   //? 2-,5-,8-,... RUNNIGN GRAPH STREAM HERE
   let inputs = {
      question: docContent[0],
   }
   console.log('\nQuery: ', chalk.red(question))
   result = await graph.invoke(inputs)
   console.log(
      '\n',
      chalk.bgGreen.bold.black(' Final Respone :'),
      chalk.bold.greenBright(result['answer'])
   )

   //! Logs Entry
   const logEntry = {
      function: 'Chatbot_HyDE',
      timestamp: new Date().toISOString(),
      website: web_url,
      originalQuestion: question,
      hypotheticalDoc: docContent[0],
      RESPONSE: result['answer'],
   }
   fs.appendFileSync('rag_logs.json', JSON.stringify(logEntry) + '\n')
   //! Formatting Logs
   exec('node logsFormatter.js', (error, stdout, stderr) => {
      if (error) {
         console.error(`Error executing script: ${error.message}`)
         return
      }
      if (stderr) {
         console.error(`Script error: ${stderr}`)
         return
      }
      console.log(`${stdout}`)
   })
   //* RETRIEVING DONE
}

const rl = readline.createInterface({ input, output })
const web_url = await rl.question(
   chalk.bold.blue`\nEnter the URL of the website: `
)

const question = await rl.question(chalk.bold.blue`\nEnter your question: `)
console.log('\n')
rl.close()
main(web_url, question)
