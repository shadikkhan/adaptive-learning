"""
Agent Graph Visualization Script
Generates visual representations of the multi-agent workflow
"""
from graph import learning_graph
from PIL import Image
import io


def generate_mermaid_diagram():
    """
    Generate a Mermaid diagram of the agent graph
    """
    try:
        # Get the Mermaid diagram as PNG
        mermaid_png = learning_graph.get_graph().draw_mermaid_png()
        
        # Save to file
        with open("agent_graph.png", "wb") as f:
            f.write(mermaid_png)
        
        print("✅ Graph visualization saved as 'agent_graph.png'")
        
        # Also get ASCII representation for terminal
        print("\n" + "="*60)
        print("AGENT WORKFLOW GRAPH")
        print("="*60)
        
        # Get Mermaid syntax
        mermaid_syntax = learning_graph.get_graph().draw_mermaid()
        print("\nMermaid Diagram Syntax:")
        print("-"*60)
        print(mermaid_syntax)
        print("-"*60)
        
        return mermaid_png
        
    except Exception as e:
        print(f"❌ Error generating Mermaid PNG: {e}")
        print("\nTrying ASCII representation instead...")
        try:
            ascii_graph = learning_graph.get_graph().draw_ascii()
            print("\n" + "="*60)
            print("AGENT WORKFLOW GRAPH (ASCII)")
            print("="*60)
            print(ascii_graph)
            return ascii_graph
        except Exception as e2:
            print(f"❌ Error generating ASCII: {e2}")
            return None


def print_graph_summary():
    """
    Print a summary of the graph structure
    """
    print("\n" + "="*60)
    print("MULTI-AGENT SYSTEM SUMMARY")
    print("="*60)
    
    print("\n📊 AGENTS (Nodes):")
    agents = [
        ("infer_intent", "Intent Classification", "Determines user intent"),
        ("simplify", "Age-Adaptive Explanation", "Simplifies content for age"),
        ("generate_example", "Example Generation", "Creates relevant examples"),
        ("think", "Reflection Question", "Generates thought-provoking questions"),
        ("quiz", "Quiz Generation", "Creates quiz questions"),
        ("save_answer", "Answer Storage", "Saves user quiz answers"),
        ("evaluate_answer", "Answer Evaluation", "Scores quiz responses"),
        ("answer_feedback", "Answer Feedback", "Provides answer feedback"),
        ("safety", "Safety Validation", "Checks age-appropriate content"),
        ("format", "Response Formatting", "Structures final output"),
    ]
    
    for i, (name, title, desc) in enumerate(agents, 1):
        print(f"  {i:2d}. {name:20s} - {title:30s} | {desc}")
    
    print("\n🔀 WORKFLOWS:")
    print("  1. Teaching Flow:")
    print("     infer_intent → simplify → generate_example → safety → think → format")
    
    print("\n  2. Quiz Generation Flow:")
    print("     infer_intent → quiz → safety → format")
    
    print("\n  3. Answer Evaluation Flow:")
    print("     infer_intent → save_answer → evaluate_answer → answer_feedback → safety → format")
    
    print("\n🎯 CONDITIONAL ROUTING:")
    print("  • Intent-based routing after 'infer_intent'")
    print("  • Context-aware routing after 'safety'")
    
    print("\n📍 ENTRY/EXIT:")
    print("  • Entry Point: infer_intent")
    print("  • Exit Point: format")
    
    print("="*60 + "\n")


def generate_html_visualization():
    """
    Generate an HTML file with embedded Mermaid diagram
    """
    try:
        mermaid_syntax = learning_graph.get_graph().draw_mermaid()
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgeXplain - Agent Workflow Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .legend {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .legend h3 {{
            margin-top: 0;
            color: #333;
        }}
        .agent-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .agent-item {{
            padding: 10px;
            background: white;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }}
        .agent-name {{
            font-weight: bold;
            color: #667eea;
        }}
        .mermaid {{
            text-align: center;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .workflows {{
            margin: 20px 0;
        }}
        .workflow-item {{
            background: #e3f2fd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #2196f3;
        }}
        .workflow-title {{
            font-weight: bold;
            color: #1976d2;
            margin-bottom: 5px;
        }}
        code {{
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #d63384;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AgeXplain Multi-Agent System</h1>
        <p class="subtitle">Interactive Agent Workflow Visualization</p>
        
        <div class="legend">
            <h3>📊 System Overview</h3>
            <p><strong>Total Agents:</strong> 10 specialized agents</p>
            <p><strong>Architecture:</strong> LangGraph State Machine</p>
            <p><strong>Purpose:</strong> Age-adaptive educational content generation</p>
        </div>
        
        <h2>Agent Workflow Diagram</h2>
        <div class="mermaid">
{mermaid_syntax}
        </div>
        
        <div class="legend">
            <h3>🔧 Agent Descriptions</h3>
            <div class="agent-list">
                <div class="agent-item">
                    <div class="agent-name">1. infer_intent</div>
                    <div>Classifies user intent (new question, followup, quiz, answer)</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">2. simplify</div>
                    <div>Creates age-appropriate explanations</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">3. generate_example</div>
                    <div>Generates relevant real-world examples</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">4. think</div>
                    <div>Creates reflective thinking questions</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">5. quiz</div>
                    <div>Generates multiple-choice quiz questions</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">6. save_answer</div>
                    <div>Stores user quiz responses</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">7. evaluate_answer</div>
                    <div>Scores quiz answers and provides evaluation</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">8. answer_feedback</div>
                    <div>Generates detailed feedback on quiz performance</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">9. safety</div>
                    <div>Validates content is age-appropriate and safe</div>
                </div>
                <div class="agent-item">
                    <div class="agent-name">10. format</div>
                    <div>Structures final response with proper formatting</div>
                </div>
            </div>
        </div>
        
        <div class="legend workflows">
            <h3>🔀 Workflow Paths</h3>
            
            <div class="workflow-item">
                <div class="workflow-title">📚 Teaching Flow (new_question/followup)</div>
                <code>infer_intent → simplify → generate_example → safety → think → format</code>
            </div>
            
            <div class="workflow-item">
                <div class="workflow-title">📝 Quiz Generation Flow</div>
                <code>infer_intent → quiz → safety → format</code>
            </div>
            
            <div class="workflow-item">
                <div class="workflow-title">✅ Answer Evaluation Flow</div>
                <code>infer_intent → save_answer → evaluate_answer → answer_feedback → safety → format</code>
            </div>
        </div>
        
        <div class="legend">
            <h3>🎯 Key Features</h3>
            <ul>
                <li><strong>Conditional Routing:</strong> Intent-based and context-aware path selection</li>
                <li><strong>State Preservation:</strong> Maintains context across agent transitions</li>
                <li><strong>Safety Layer:</strong> Every response validated for age-appropriateness</li>
                <li><strong>Modular Design:</strong> Each agent is independently testable and upgradeable</li>
            </ul>
        </div>
    </div>
    
    <script>
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                curve: 'basis',
                useMaxWidth: true
            }}
        }});
    </script>
</body>
</html>"""
        
        with open("agent_graph.html", "w") as f:
            f.write(html_content)
        
        print("✅ HTML visualization saved as 'agent_graph.html'")
        print("   Open this file in a browser to see the interactive diagram!")
        
    except Exception as e:
        print(f"❌ Error generating HTML visualization: {e}")


if __name__ == "__main__":
    print("\n🎨 Generating Agent Workflow Visualizations...\n")
    
    # Print summary
    print_graph_summary()
    
    # Generate visualizations
    print("\n📊 Generating visual diagrams...\n")
    
    # Try to generate Mermaid PNG
    generate_mermaid_diagram()
    
    # Generate HTML visualization
    generate_html_visualization()
    
    print("\n✨ Visualization complete!")
    print("\nGenerated files:")
    print("  • agent_graph.png  - PNG image of the workflow")
    print("  • agent_graph.html - Interactive HTML visualization")
    print("\n💡 Tip: Open agent_graph.html in your browser for the best experience!")
