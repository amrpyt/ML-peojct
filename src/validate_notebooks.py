import json
import os
from pprint import pprint

def read_notebook(filename):
    """Read a Jupyter notebook file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return None

def get_section_structure(notebook):
    """Extract section structure from notebook."""
    sections = []
    current_section = None
    required_imports = {
        'tensorflow': False,
        'numpy': False,
        'pandas': False,
        'psutil': False
    }
    
    for cell in notebook.get('cells', []):
        # Check imports
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']).lower()
            for lib in required_imports:
                if f'import {lib}' in source or f'from {lib}' in source:
                    required_imports[lib] = True
        
        # Check sections
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source']).lower()
            if source.startswith('#') and not source.startswith('##'):
                title = source.lstrip('#').strip()
                current_section = {
                    'title': title,
                    'level': 1,
                    'cell_count': 0,
                    'code_cells': 0,
                    'markdown_cells': 0,
                    'imports': required_imports.copy()
                }
                sections.append(current_section)
        
        if current_section is not None:
            current_section['cell_count'] += 1
            if cell['cell_type'] == 'code':
                current_section['code_cells'] += 1
            else:
                current_section['markdown_cells'] += 1
    
    return sections

def validate_notebooks():
    """Validate all notebooks in sequence."""
    notebook_files = [
        ('Data Loading and Preprocessing', 'src/01_data_preprocessing.ipynb'),
        ('Model Definitions', 'src/02_model_definitions.ipynb'),
        ('Model Training', 'src/03_model_training.ipynb'),
        ('Model Performance Analysis', 'src/model_metrics.ipynb'),
        ('Model Optimization', 'src/04_model_optimization.ipynb'),
        ('Temperature and Humidity Prediction', 'src/05_temp_hum_prediction.ipynb'),
        ('Visualization and Comparison', 'src/06_visualization.ipynb')
    ]
    
    required_sections = {
        'data loading and preprocessing': ['data preprocessing', 'outlier detection', 'feature normalization'],
        'model definitions': ['cnn-lstm', 'cnn-bilstm', '1dcnn', 'rnn', 'lstm'],
        'model training': ['training', 'evaluation', 'metrics'],
        'model performance analysis': ['efficiency', 'memory usage', 'inference time'],
        'model optimization': ['optimization', 'quantization'],
        'temperature and humidity prediction': ['temperature', 'humidity'],
        'visualization and comparison': ['comparison', 'visualization']
    }
    
    found_sections = set()
    validation_results = {}
    
    print("Validating notebook sections...")
    print("-" * 50)
    
    # Check each notebook exists
    missing_files = []
    for title, filepath in notebook_files:
        if not os.path.exists(filepath):
            missing_files.append(filepath)
    
    if missing_files:
        print("\nMissing notebook files:")
        for file in missing_files:
            print(f"- {file}")
        return False
    
    # Validate each notebook
    for title, filepath in notebook_files:
        print(f"\nChecking {filepath}...")
        notebook = read_notebook(filepath)
        if notebook:
            sections = get_section_structure(notebook)
            if sections:
                main_title = sections[0]['title'].lower()
                found_sections.add(main_title)
                
                # Check required content
                required = required_sections.get(main_title, [])
                found_content = set()
                for cell in notebook['cells']:
                    content = ''.join(cell['source']).lower()
                    for req in required:
                        if req in content:
                            found_content.add(req)
                
                validation_results[title] = {
                    'file': filepath,
                    'main_title': main_title,
                    'total_cells': len(notebook.get('cells', [])),
                    'sections': sections,
                    'required_content': {
                        'found': list(found_content),
                        'missing': list(set(required) - found_content)
                    }
                }
                
                print(f"Found main section: {sections[0]['title']}")
                print(f"Total cells: {validation_results[title]['total_cells']}")
                if validation_results[title]['required_content']['missing']:
                    print("Missing required content:")
                    for item in validation_results[title]['required_content']['missing']:
                        print(f"  - {item}")
    
    # Print detailed validation report
    print("\nValidation Report:")
    print("-" * 50)
    
    all_required = set(title.lower() for title, _ in notebook_files)
    missing_sections = all_required - found_sections
    
    if missing_sections:
        print("\nMissing sections:")
        for section in missing_sections:
            print(f"- {section}")
    else:
        print("\nAll required sections found!")
    
    print("\nDetailed Section Analysis:")
    for title, result in validation_results.items():
        print(f"\n{title}:")
        print(f"File: {result['file']}")
        print(f"Main title: {result['main_title']}")
        print(f"Total cells: {result['total_cells']}")
        if result['required_content']['missing']:
            print("Missing content:")
            for item in result['required_content']['missing']:
                print(f"  - {item}")
    
    # Check if any sections are missing required content
    has_missing_content = any(
        result['required_content']['missing']
        for result in validation_results.values()
    )
    
    return len(missing_sections) == 0 and not has_missing_content

if __name__ == "__main__":
    success = validate_notebooks()
    print("\nValidation " + ("successful!" if success else "failed."))