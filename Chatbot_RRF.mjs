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
import { SortXYZBlockchainLoader } from '@langchain/community/document_loaders/web/sort_xyz_blockchain'

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

   //? CHATPROMPT TEMPLATE INSTANTIATION -working below one, just commented bcz it was for checkign with example its working

   const promptTemplate = await pull('rlm/rag-prompt')

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

   //? RETRIEVAL PROCESS - RRF based
   const retrieve = async (state) => {
      const baseQuestion = state.question
      const queryGeneratorPrompt = `Generate 3 different but related search queries based on this question:\n "${baseQuestion}"
      Respond ONLY with a raw JSON array, no explanation, no code block, no backticks. Example: ["query1", "query2", "query3"]`
      const queryResponse = await llm.invoke(queryGeneratorPrompt)
      let queryVariations
      try {
         queryVariations = JSON.parse(queryResponse.content)
      } catch (error) {
         console.log('Error parsing LLM generated queries:', error)
         return
      }

      const retrievalPromises = queryVariations.map((query) =>
         vectorStore.similaritySearch(query, 2)
      )
      const resultsArrays = await Promise.all(retrievalPromises)

      //    //? Main Logic of Optimization Algo here - not working below code
      //    const allResults = resultsArrays.flat()
      //    const prioritizedResults = reciprocalRankFusion(allResults).top(4)
      //    return { context: prioritizedResults }
      // }

      // const reciprocalRankFusion = (results, k = 60) => {
      //    scores = {}

      //    for (let result of results) {
      //       for(let r, doc_id in Enumerator(result)) {
      //         scores[doc_id] = scores.get(doc_id, 0) + 1 / (r + k +1)

      //       }
      //    }
      //    return Sorted(scores.items(), key=lambda x: x[1], reverse=True)
      // }

      const allResults = resultsArrays.flat()

      const uniqueResults = Array.from(
         new Map(
            allResults.map((doc) => [
               doc.metadata?.source || doc.pageContent,
               doc,
            ])
         ).values()
      )

      const prioritizedResults = reciprocalRankFusion(uniqueResults).top(4)
      console.log('\nPrioritized results after RF:\n', prioritizedResults)
      return { context: prioritizedResults }
   }

   const reciprocalRankFusion = (results, k = 60) => {
      const scores = {}

      results.forEach((doc, r) => {
         // Now 'results' is already unique
         const doc_id = doc.metadata?.source || doc.pageContent
         scores[doc_id] = (scores[doc_id] || 0) + 1 / (r + k + 1)
      })

      const sortedScores = Object.entries(scores).sort(
         ([, scoreA], [, scoreB]) => scoreB - scoreA
      )

      return {
         top: (n) =>
            sortedScores
               .slice(0, n)
               .map(([key]) =>
                  results.find(
                     (doc) => (doc.metadata?.source || doc.pageContent) === key
                  )
               ),
      }
   }

   //* GEneration Process

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
      // console.log(`Answer:` + chalk.bold.greenBright + ` ${result['answer']}`)
      console.log(`\nAnswer: ` + chalk.bold.greenBright`${result['answer']}`)
      spinnerStream.succeed('Response Retrieved.')
   } catch (error) {
      console.log('Error calling graph.stream():', error)
      spinnerStream.fail('Failed to Retrieve.')
   }
}

const rl = readline.createInterface({ input, output })
const web_url = await rl.question(
   chalk.bold.blue`\nEnter the URL of the website: `
)

const question = await rl.question(chalk.bold.blue`\nEnter your question: `)
console.log('\n')
rl.close()
main(web_url, question)
