// Import necessary modules from LangChain
import "pdf-parse"
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import "dotenv/config"; // Load environment variables (e.g., OpenAI API key)
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"

// Initialize the OpenAI model with API key and model type
const openaiModel = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-4o-mini",
});

// Step 1: Load Web Page Content
// Using CheerioWebBaseLoader to scrape content from a webpage
// const loader = new CheerioWebBaseLoader(
//   "https://circleci.com/blog/introduction-to-graphql/", // Webpage URL
//   {} // Additional options (if any)
// );
const loader = new PDFLoader("./myPDF.pdf")

const documents = await loader.load(); // Load the webpage content as documents

// Step 2: Split the Text into Chunks
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1200,  // Maximum size of each text chunk
  chunkOverlap: 200, // Overlap between chunks to retain context
});

const pdfPageSplits = await textSplitter.splitDocuments(documents); // Split document into smaller chunks

// Step 3: Embed and Store in a Vector Database
const embedder = new OpenAIEmbeddings(); // Initialize OpenAI embeddings model
const store = await MemoryVectorStore.fromDocuments(pdfPageSplits, embedder); // Store embedded chunks in memory

// Step 4: Retrieve Relevant Documents Based on a Query
const pdfPageRetrieval = store.asRetriever(); // Convert vector store into a retriever
const relevantDocs = await pdfPageRetrieval.invoke("How does GraphQL work"); // Retrieve relevant documents based on query

// Step 5: Define a Prompt for Answer Generation
const instructionPrompt = `You are an assistant, and your role is to answer the questions.
Use the following pieces of retrieved context to answer the question.
Keep the answer concise.
{context}
`;

// Define a Retrieval-Augmented Generation (RAG) prompt
const ragPrompt = ChatPromptTemplate.fromMessages([
  ["system", instructionPrompt], // System-level instruction prompt
  ["user", "{input}"], // Placeholder for user input (question)
]);

// Step 6: Create a Chain for Answer Generation
const questionAnswerChain = await createStuffDocumentsChain({
  llm: openaiModel, // Use the initialized OpenAI model
  prompt: ragPrompt, // Use the defined RAG prompt
});

// Step 7: Create a Retrieval-Augmented Chain
const ragChain = await createRetrievalChain({
  retriever: pdfPageRetrieval, // Use the document retriever
  combineDocsChain: questionAnswerChain, // Combine retrieved documents with the answer generation chain
});

// Step 8: Invoke the RAG Chain with a Question
const response = await ragChain.invoke({
  input: "How is GraphQL compared to REST", // User query
});

// Step 9: Print the Final Answer
console.log(response.answer);
