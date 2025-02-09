import json
import networkx as nx
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline, AutoTokenizer
import matplotlib.pyplot as plt
from collections import defaultdict
class VehicleDataProcessor:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./vehicle_database")
        try:
            self.collection = self.chroma_client.get_collection("vehicle_data")
        except:
            self.collection = self.chroma_client.create_collection(
                name="vehicle_data",
                metadata={"hnsw:space": "cosine"}
            )
        
        self.graph = nx.DiGraph()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def process_json_data(self, json_file_path):
        try:
            existing_count = len(self.collection.get()['ids'])
            if existing_count > 0:
                print("Data already exists in ChromaDB. Skipping processing.")
                return
        except:
            pass

        print("Processing new data...")
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        vehicle_data = defaultdict(lambda: {'confidence': -1})
        for entry in data:
            vid = entry['vehicle_id']
            if entry['confidence'] > vehicle_data[vid]['confidence']:
                vehicle_data[vid] = entry

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for vid, entry in vehicle_data.items():
            # Include vehicle_id in metadata
            metadata = {
                'vehicle_id': vid,
                'plate': entry['plate'],
                'frame': entry['frame'],
                'confidence': entry['confidence'],
                'brand': entry.get('brand', 'unknown')
            }
            
            doc_text = f"Vehicle {vid} with license plate {entry['plate']} "
            doc_text += f"is a {entry.get('brand', 'unknown')} brand vehicle "
            doc_text += f"detected in frame {entry['frame']} with {entry['confidence']}% confidence."

            embedding = self.embedding_model.encode(doc_text)

            documents.append(doc_text)
            embeddings.append(embedding.tolist())
            metadatas.append(metadata)
            ids.append(f"vehicle_{vid}")

            # Knowledge graph processing
            vehicle_node = f"vehicle_{vid}"
            plate_node = f"plate_{entry['plate']}"
            brand_node = f"brand_{entry.get('brand', 'unknown')}"

            self.graph.add_node(vehicle_node, 
                              type='vehicle',
                              frame=entry['frame'],
                              confidence=entry['confidence'],
                              plate=entry['plate'],
                              brand=entry.get('brand', 'unknown'),
                              vehicle_id=vid)
            
            self.graph.add_node(plate_node, 
                              type='plate',
                              plate_number=entry['plate'])
            
            self.graph.add_node(brand_node, 
                              type='brand',
                              brand_name=entry.get('brand', 'unknown'))

            self.graph.add_edge(vehicle_node, plate_node, relation='has_plate')
            self.graph.add_edge(vehicle_node, brand_node, relation='has_brand')

        if documents:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

        nx.write_graphml(self.graph, 'vehicle_knowledge_graph.graphml')
        print(f"Processed {len(vehicle_data)} unique vehicles.")
class VehicleQuerySystem:
    def __init__(self, graph_path='vehicle_knowledge_graph.graphml'):
        self.graph = nx.read_graphml(graph_path)
        self.chroma_client = chromadb.PersistentClient(path="./vehicle_database")
        self.collection = self.chroma_client.get_collection("vehicle_data")

        self.model_id = "meta-llama/Llama-3.2-1B"
        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    def answer_question(self, question):
        """Answer questions about vehicles using RAG."""
        import re
        from difflib import SequenceMatcher

        def extract_metadata_summary(metadata_list):
            """Create a comprehensive summary of vehicle information."""
            summary = {}
            for metadata in metadata_list:
                vid = metadata['vehicle_id']
                if vid not in summary or metadata['confidence'] > summary[vid]['confidence']:
                    summary[vid] = metadata
            return summary

        def find_similar_plates(query_plate, all_plates, threshold=0.6):
            similar = []
            for metadata in all_plates:
                plate = metadata.get('plate', '')
                if plate:
                    direct_similarity = SequenceMatcher(None, query_plate.upper(), plate.upper()).ratio()
                    clean_query = re.sub(r'[^A-Z0-9]', '', query_plate.upper())
                    clean_plate = re.sub(r'[^A-Z0-9]', '', plate.upper())
                    clean_similarity = SequenceMatcher(None, clean_query, clean_plate).ratio()
                    similarity = max(direct_similarity, clean_similarity)
                    if similarity > threshold:
                        similar.append((metadata, similarity))
            return sorted(similar, key=lambda x: x[1], reverse=True)

        # Get all metadata
        all_metadata = self.collection.get()['metadatas']
        metadata_summary = extract_metadata_summary(all_metadata)

        # Build comprehensive context based on the question
        context = "Current traffic analysis summary:\n"
        
        # Check question type
        question_lower = question.lower()
        
        # Extract potential plate number
        plate_match = re.search(r'[A-Z0-9]{3,}', question.upper())
        query_plate = plate_match.group(0) if plate_match else None

        if query_plate:
            # Handle license plate queries
            exact_matches = [m for m in all_metadata if m.get('plate', '').upper() == query_plate.upper()]
            if exact_matches:
                context += f"\nFound vehicle(s) with plate {query_plate}:\n"
                for metadata in exact_matches:
                    context += f"- Vehicle {metadata['vehicle_id']}: {metadata['plate']} "
                    context += f"({metadata.get('brand', 'unknown')} brand) "
                    context += f"in frame {metadata['frame']} "
                    context += f"with {metadata['confidence']}% confidence\n"
            else:
                similar = find_similar_plates(query_plate, all_metadata)
                if similar:
                    context += f"\nNo exact match for {query_plate}, but found similar plates:\n"
                    for metadata, similarity in similar[:3]:
                        context += f"- Vehicle {metadata['vehicle_id']}: {metadata['plate']} "
                        context += f"(similarity: {similarity:.2f})\n"

        # Add relevant information based on question keywords
        if 'brand' in question_lower or 'make' in question_lower:
            context += "\nVehicle brands detected:\n"
            brands = {}
            for metadata in metadata_summary.values():
                brand = metadata.get('brand', 'unknown')
                if brand not in brands:
                    brands[brand] = []
                brands[brand].append(metadata['vehicle_id'])
            for brand, vehicles in brands.items():
                context += f"- {brand}: Vehicles {', '.join(map(str, vehicles))}\n"

        if 'frame' in question_lower:
            context += "\nFrame information:\n"
            frame_data = sorted([(m['frame'], m['vehicle_id'], m.get('plate', 'unknown')) 
                            for m in metadata_summary.values()])
            for frame, vid, plate in frame_data:
                context += f"- Frame {frame}: Vehicle {vid} (plate: {plate})\n"

        if 'confidence' in question_lower or 'accuracy' in question_lower:
            context += "\nConfidence levels:\n"
            conf_data = sorted([(m['confidence'], m['vehicle_id'], m.get('plate', 'unknown')) 
                            for m in metadata_summary.values()], reverse=True)
            for conf, vid, plate in conf_data:
                context += f"- Vehicle {vid} (plate: {plate}): {conf:.2f}% confidence\n"

        # Add graph relationships for more context
        for vid in metadata_summary.keys():
            vehicle_node = f"vehicle_{vid}"
            if self.graph.has_node(vehicle_node):
                neighbors = list(self.graph.neighbors(vehicle_node))
                related_info = []
                for neighbor in neighbors:
                    if neighbor.startswith('plate_'):
                        related_info.append(f"plate: {self.graph.nodes[neighbor]['plate_number']}")
                    elif neighbor.startswith('brand_'):
                        related_info.append(f"brand: {self.graph.nodes[neighbor]['brand_name']}")
                if related_info:
                    context += f"\nVehicle {vid} relationships: {', '.join(related_info)}\n"

        prompt = f"""You are an AI assistant analyzing traffic surveillance data. 
    Based on the following comprehensive information (which should not be included in your response),
    provide a natural, detailed answer to the user's question. If the user asks about a specific plate
    and it's not found, suggest similar plates. If the question is about statistics or patterns,
    include relevant numbers and trends. Always be specific and precise in your answer.

    Context:
    {context}

    Question: {question}
    Answer: """

        response = self.pipe(
            prompt,
            max_new_tokens=300,  # Increased for more detailed responses
            truncation=True,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        answer_text = response[0]['generated_text']
        if "Answer:" in answer_text:
            answer_text = answer_text.split("Answer:")[1].strip()
        return answer_text
    def visualize_graph(self):
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.graph)
        
        vehicle_nodes = [n for n in self.graph.nodes() if n.startswith('vehicle_')]
        plate_nodes = [n for n in self.graph.nodes() if n.startswith('plate_')]
        brand_nodes = [n for n in self.graph.nodes() if n.startswith('brand_')]
        
        nx.draw_networkx_nodes(self.graph, pos, nodelist=vehicle_nodes, node_color='lightblue', node_size=500)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=plate_nodes, node_color='lightgreen', node_size=500)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=brand_nodes, node_color='lightpink', node_size=500)
        
        nx.draw_networkx_edges(self.graph, pos)
        
        labels = {node: node.split('_')[1] for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels)
        
        plt.title("Vehicle Knowledge Graph")
        plt.axis('off')
        return plt.gcf()
def main():
    # Process data
    processor = VehicleDataProcessor()
    processor.process_json_data('tracking_results_brand_2.json')

    # Query system
    query_system = VehicleQuerySystem()

    while True:
        print("\nOptions:")
        print("1. Ask a question")
        print("2. Visualize knowledge graph")
        print("3. Quit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == '1':
            question = input("Enter your question: ")
            answer = query_system.answer_question(question)
            print(f"\nAnswer: {answer}")

        elif choice == '2':
            fig = query_system.visualize_graph()
            plt.show()

        elif choice == '3':
            break

        else:
            print("Invalid choice. Please try again.")
if __name__ == "__main__":
    main()