const fs = require('fs')
const path = require('path')

// Input file (original raw logs)
const inputFilePath = path.join(__dirname, 'rag_logs.json')

// Output file (formatted logs)
const outputFilePath = path.join(__dirname, 'rag_logs.yaml')

try {
   // Read raw logs
   const rawData = fs.readFileSync(inputFilePath, 'utf-8')

   // Split logs by lines, parse each JSON line
   const logs = rawData
      .split('\n')
      .filter(Boolean)
      .map((line) => JSON.parse(line))

   // Build formatted string
   const formattedLogs = logs
      .map((log) => {
         const entries = Object.entries(log)
            .map(([key, value]) => `${key}: ${value}`)
            .join('\n')

         return `${entries}\n---------- LOG END ----------`
      })
      .join('\n\n') // Two newlines between logs

   // Write to the new output file
   fs.appendFileSync(outputFilePath, formattedLogs)

   console.log('\n✅ Logs saved to rag_logs.yaml!\n')

   // Delete old file
   fs.unlinkSync(inputFilePath)
   // console.log('🧹 Original rag_logs.json deleted!')
} catch (error) {
   console.error('\n❌ Error processing logs:', error)
}

//? SMALLER VERSION CODE BELOW BUT NEED small Package INSTALL

// npm install js-yaml //DO THIS FIRST

// import fs from 'fs'
// import yaml from 'js-yaml'

// const raw = fs
//    .readFileSync('rag_logs.json', 'utf-8')
//    .split('\n')
//    .filter(Boolean)
//    .map((line) => JSON.parse(line))

// const yamlData = yaml.dump(raw)

// fs.writeFileSync('rag_logs.yaml', yamlData)

// // Delete old file
// fs.unlinkSync('rag_logs.json')

// console.log('✅ JSON converted to YAML and original JSON deleted!')
