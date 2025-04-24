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
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf'

dotenv.config()

async function main(type, web_url, question) {
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
         collectionName: 'langchainjs-testing2',
      }
   )

   //* INSTANTIATING PROCESS Done

   //* INDEXING PROCESS START

   const spinnerIndex = ora('Indexing Process...\n').start()
   let docs
   try {
      if (type === 'web') {
         //? Document Loading using CHEERIO
         const pTagSelector = 'p'
         const cheerioLoader = new CheerioWebBaseLoader(web_url, {
            selector: pTagSelector,
         })

         docs = await cheerioLoader.load()
      } else if (type === 'pdf') {
         const loader = new PDFLoader(web_url)
         docs = await loader.load()
      }
      // console.log('docs :', docs)
      if (memo === 'memory') {
         docs.forEach((doc) => {
            if (!doc.metadata) doc.metadata = {}
            doc.metadata.memory = true
         })
      }

      console.log('docs after pushing ', docs)

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

   const promptTemplate = await pull('rlm/rag-prompt')

   //
   // * USING LANGGRAPH TO WRAP EVERYTHING INSIDE ONE FLOW

   //? States for langgraph - contains questions and responses at each step (NODES)
   // import { Document } from '@langchain/core/documents'

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

   let inputs = {
      question: question,
   }

   //? Simple working
   const spinnerStream = ora('Calling graph.stream()...').start()
   try {
      const result = await graph.invoke(inputs)
      // console.log(result.context.slice(0, 2))
      console.log('\n')
      // console.log(`Answer:` + chalk.bold.greenBright + ` ${result['answer']}`)
      console.log(`\nAnswer: ` + chalk.bold.greenBright`${result['answer']}`)
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
const type = await rl.question(
   chalk.bold.blue`\nEnter the type of Data source (web or pdf): `
)
const memo = await rl.question(
   chalk.bold
      .blue`\nYou want to memorize this data or not ( for yes type 'memory' else 'no'): `
)
const web_url = await rl.question(
   chalk.bold.blue`\nEnter the URL of the website: `
)

const question = await rl.question(chalk.bold.blue`\nEnter your question: `)
console.log('\n')
rl.close()
main(type, memo, web_url, question)
