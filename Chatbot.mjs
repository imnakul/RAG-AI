import dotenv from 'dotenv'
import { ChatGoogleGenerativeAI } from '@langchain/google-genai'
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai'
import { QdrantVectorStore } from '@langchain/qdrant'

dotenv.config()

//? Instantiating GEMINI CHAT
const llm = new ChatGoogleGenerativeAI({
   model: 'gemini-2.0-flash',
   temperature: 0,
   // maxRetries: 2,
   // other params...
})

//? Embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
   model: 'text-embedding-3-large', // 768 dimensions
})

//? Vector Store

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
   url: process.env.QDRANT_URL,
   collectionName: 'langchainjs-testing',
})

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
