import cv2
import numpy as np
import json
import os
import glob
from typing import List, Dict, Any
import importlib.util
import sys

class VideoChunkProcessor:
    """
    Processes video chunks by applying filters and transitions based on JSON modifications.
    Can import functions from .py files or .ipynb notebooks.
    """
    
    def __init__(self, chunks_folder: str, modifications_json: str, output_video_path: str,
                 filters_file: str = "filters.py", transitions_file: str = "transitions.py"):
        """
        Initialize the processor.
        
        Args:
            chunks_folder: Folder containing video chunks (chunk_0.mp4, chunk_1.mp4, etc.)
            modifications_json: Path to JSON file with modifications
            output_video_path: Path for final output video
            filters_file: Path to filters file (.py or .ipynb)
            transitions_file: Path to transitions file (.py or .ipynb)
        """
        self.chunks_folder = chunks_folder
        self.output_video_path = output_video_path
        self.filters_file = filters_file
        self.transitions_file = transitions_file
        
        # Load modifications from JSON file
        with open(modifications_json, 'r') as f:
            self.modifications = json.load(f)
        
        # Get list of chunk files
        self.chunk_files = sorted(glob.glob(os.path.join(chunks_folder, "chunk_*.mp4")))
        self.total_chunks = len(self.chunk_files)
        
        # Load filter and transition modules
        self.filters_module = self.load_module_from_file(filters_file, "filters")
        self.transitions_module = self.load_module_from_file(transitions_file, "transitions")
        
        print(f"Found {self.total_chunks} chunks in {chunks_folder}")
        print(f"Modifications to apply: {len(self.modifications.get('modifications', []))}")

    def load_module_from_file(self, file_path: str, module_name: str):
        """Load a module from either .py or .ipynb file."""
        try:
            if file_path.endswith('.ipynb'):
                return self.load_from_notebook(file_path, module_name)
            elif file_path.endswith('.py'):
                return self.load_from_python_file(file_path, module_name)
            else:
                print(f"Warning: Unsupported file type for {file_path}")
                return None
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def load_from_python_file(self, file_path: str, module_name: str):
        """Load module from .py file."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def load_from_notebook(self, notebook_path: str, module_name: str):
        """Load functions from .ipynb notebook."""
        try:
            import nbformat
            from nbconvert import PythonExporter
            
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Convert to Python code
            exporter = PythonExporter()
            python_code, _ = exporter.from_notebook_node(notebook)
            
            # Create module and execute code
            module = type(sys)('notebook_module')
            exec(python_code, module.__dict__)
            
            return module
            
        except ImportError:
            print("Warning: nbformat and nbconvert required for .ipynb files")
            print("Install with: pip install nbformat nbconvert")
            return None
        except Exception as e:
            print(f"Error loading notebook {notebook_path}: {e}")
            return None

    def load_chunk_frames(self, chunk_path: str) -> List[np.ndarray]:
        """Load all frames from a chunk video file."""
        cap = cv2.VideoCapture(chunk_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames

    def get_video_properties(self, chunk_path: str) -> tuple:
        """Get video properties (width, height, fps) from a chunk."""
        cap = cv2.VideoCapture(chunk_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return width, height, fps

    def apply_filter_to_chunk(self, chunk_frames: List[np.ndarray], filter_name: str) -> List[np.ndarray]:
        """Apply filter to chunk frames by calling the appropriate function."""
        print(f"    Applying filter: {filter_name}")
        
        if self.filters_module is None:
            print("    Warning: Filters module not loaded, keeping original frames")
            return chunk_frames
        
        try:
            # Get filter function
            filter_func = self.get_filter_function(filter_name)
            
            if filter_func is None:
                print(f"    Warning: Filter '{filter_name}' not found, keeping original frames")
                return chunk_frames
            
            # Apply filter to each frame
            filtered_frames = []
            for frame in chunk_frames:
                filtered_frame = filter_func(frame)
                filtered_frames.append(filtered_frame)
            
            return filtered_frames
            
        except Exception as e:
            print(f"    Error applying filter '{filter_name}': {e}")
            return chunk_frames

    def get_filter_function(self, filter_name: str):
        """Get the filter function by name."""
        if self.filters_module is None:
            return None
        
        # Map filter names to function names
        filter_mapping = {
            'grayscale': 'grayscale',
            'invert_colors': 'invert_colors', 
            'sepia': 'sepia',
            'pencil_sketch': 'pencil_sketch',
            'cartoon_effect': 'cartoon_effect',
            'edge_detection': 'edge_detection',
            'gaussian_blur': 'gaussian_blur',
            'motion_blur': 'motion_blur',
            'emboss_filter': 'emboss_filter',
            'color_tint': 'color_tint',
            'sobel_x': 'sobel_x',
            'sobel_y': 'sobel_y',
            'laplacian': 'laplacian',
            'median_blur': 'median_blur',
            'bilateral_filter': 'bilateral_filter',
            'sharpen': 'sharpen',
            'hsv_filter': 'hsv_filter',
            'negative_hsv': 'negative_hsv',
            'thresholding': 'thresholding',
            'adaptive_threshold': 'adaptive_threshold',
            'dilation': 'dilation'
        }
        
        if filter_name in filter_mapping:
            func_name = filter_mapping[filter_name]
            if hasattr(self.filters_module, func_name):
                return getattr(self.filters_module, func_name)
        
        return None

    def create_transition_between_chunks(self, chunk1_frames: List[np.ndarray], 
                                       chunk2_frames: List[np.ndarray], 
                                       transition_name: str) -> List[np.ndarray]:
        """Create transition between two chunks by calling the appropriate function."""
        print(f"    Creating transition: {transition_name}")
        
        if self.transitions_module is None:
            print("    Warning: Transitions module not loaded, using simple concatenation")
            return chunk1_frames + chunk2_frames
        
        try:
            # Get transition function
            transition_func = self.get_transition_function(transition_name)
            
            if transition_func is None:
                print(f"    Warning: Transition '{transition_name}' not found, using simple concatenation")
                return chunk1_frames + chunk2_frames
            
            # Apply transition
            result_frames = transition_func(chunk1_frames, chunk2_frames)
            return result_frames
            
        except Exception as e:
            print(f"    Error applying transition '{transition_name}': {e}")
            return chunk1_frames + chunk2_frames

    def get_transition_function(self, transition_name: str):
        """Get the transition function by name."""
        if self.transitions_module is None:
            return None
        
        # Map transition names to function names
        transition_mapping = {
            'crossfade': 'crossfade_transition',
            'slide_left': 'slide_left_transition',
            'wipe_right': 'wipe_right_transition',
            'fade_to_black': 'fade_to_black_transition',
            'glitch': 'glitch_transition',
            'circle_reveal': 'circle_reveal_transition',
            'zoom_out': 'zoom_out_transition',
            'diagonal_wipe': 'diagonal_wipe_transition',
            'split_horizontal': 'split_horizontal_transition',
            'pixel_dissolve': 'pixel_dissolve_transition',
            'wave_slide': 'wave_slide_transition',
            'zoom_blur': 'zoom_blur_transition',
            'vertical_uncover': 'vertical_uncover_transition',
            'radial_wipe': 'radial_wipe_transition',
            'checkerboard': 'checkerboard_transition',
            'curtain_open': 'curtain_open_transition',
            'iris_box': 'iris_box_transition',
            'rgb_split': 'rgb_split_transition',
            'door_open': 'door_open_transition',
            'horizontal_ripple': 'horizontal_ripple_transition',
            'swirl_rotation': 'swirl_rotation_transition',
            'tile_collapse': 'tile_collapse_transition',
            'tv_static': 'tv_static_transition',
            'cross_zoom': 'cross_zoom_transition',
            'page_turn': 'page_turn_transition',
            'wave_distortion': 'wave_distortion_transition',
            'hexagon_wipe': 'hexagon_wipe_transition',
            'glitch_distortion': 'glitch_distortion_transition',
            'liquid_ink_splash': 'liquid_ink_splash_transition',
            'flip_3d_perspective': 'flip_3d_perspective_transition'
        }
        
        if transition_name in transition_mapping:
            func_name = transition_mapping[transition_name]
            if hasattr(self.transitions_module, func_name):
                return getattr(self.transitions_module, func_name)
        
        return None

    def process_chunks(self):
        """
        Seamlessly blends chunks with transitions:
        - Play chunk up to last N frames
        - Insert transition between chunks
        - Start next chunk AFTER first N frames (used in transition)
        """
        print("Starting chunk processing...")

        filters_by_chunk = {}
        transitions_by_chunk = {}

        for mod in self.modifications.get('modifications', []):
            if mod['type'] == 'filter':
                filters_by_chunk[mod['chunk']] = mod
            elif mod['type'] == 'transition':
                transitions_by_chunk[mod['chunk']] = mod

        final_video_frames = []
        transition_overlap = 15  # number of frames to blend at each transition

        chunk_id = 0
        while chunk_id < self.total_chunks:
            print(f"Processing chunk {chunk_id}")
            chunk_path = os.path.join(self.chunks_folder, f"chunk_{chunk_id}.mp4")
            if not os.path.exists(chunk_path):
                print(f"Warning: Chunk file {chunk_path} not found!")
                chunk_id += 1
                continue

            chunk_frames = self.load_chunk_frames(chunk_path)
            if chunk_id in filters_by_chunk:
                filter_mod = filters_by_chunk[chunk_id]
                print(f"  Filter: {filter_mod['name']} - {filter_mod['reason']}")
                chunk_frames = self.apply_filter_to_chunk(chunk_frames, filter_mod['name'])

            if chunk_id in transitions_by_chunk:
                transition_mod = transitions_by_chunk[chunk_id]
                target_chunk_id = transition_mod['target_chunk']
                target_chunk_path = os.path.join(self.chunks_folder, f"chunk_{target_chunk_id}.mp4")

                if not os.path.exists(target_chunk_path):
                    print(f"  Warning: Target chunk {target_chunk_path} not found!")
                    final_video_frames.extend(chunk_frames)
                    chunk_id += 1
                    continue

                target_frames = self.load_chunk_frames(target_chunk_path)
                if target_chunk_id in filters_by_chunk:
                    target_filter = filters_by_chunk[target_chunk_id]
                    print(f"    Also applying filter to target chunk: {target_filter['name']}")
                    target_frames = self.apply_filter_to_chunk(target_frames, target_filter['name'])

                # Split and prepare frames for transition
                from_main = chunk_frames[:-transition_overlap]
                from_tail = chunk_frames[-transition_overlap:]

                to_head = target_frames[:transition_overlap]
                to_rest = target_frames[transition_overlap:]

                print(f"  Transition: {transition_mod['name']} to chunk {target_chunk_id}")
                print(f"    Reason: {transition_mod['reason']}")
                transition_frames = self.create_transition_between_chunks(from_tail, to_head, transition_mod['name'])

                # Append everything properly
                final_video_frames.extend(from_main)
                final_video_frames.extend(transition_frames)

                # Next chunk: skip first N frames since used in transition
                chunk_frames = to_rest
                chunk_id = target_chunk_id
            else:
                # No transition — just append entire chunk
                final_video_frames.extend(chunk_frames)
                chunk_id += 1

        self.write_final_video(final_video_frames)
        print(f"Processing complete! Final video saved to: {self.output_video_path}")





    def write_final_video(self, frames: List[np.ndarray]):
        """Write all processed frames to final video file."""
        if not frames:
            print("No frames to write!")
            return
        
        # Get video properties from first chunk
        if self.chunk_files:
            width, height, fps = self.get_video_properties(self.chunk_files[0])
        else:
            height, width = frames[0].shape[:2]
            fps = 30
        
        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Ensure frame is correct size
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
        
        out.release()
        print(f"Written {len(frames)} frames to {self.output_video_path}")


def main():
    """Example usage and setup"""
    
    # Example JSON configuration
    sample_json = {
        "modifications": [
            {
                "chunk": 0,
                "type": "filter", 
                "name": "grayscale",
                "reason": "This segment has a serious tone. Grayscale enhances the emotional depth."
            },
            {
                "chunk": 1,
                "type": "transition",
                "target_chunk": 2,
                "name": "glitch",
                "reason": "Scene change with sudden energy shift. Glitch creates a dramatic effect."
            }
        ]
    }
    
    # Save sample JSON for testing
    with open('sample_modifications.json', 'w') as f:
        json.dump(sample_json, f, indent=2)
    
    print("✅ Enhanced VideoChunkProcessor is ready!")
    print("\n📁 Supported file structure:")
    print("├── video_processor.py (this file)")
    print("├── filters.ipynb or filters.py (your filter functions)")
    print("├── transitions.ipynb or transitions.py (your transition functions)")
    print("├── modifications.json")
    print("├── chunks/")
    print("│   ├── chunk_0.mp4")
    print("│   ├── chunk_1.mp4")
    print("│   └── ...")
    print("└── final_video.mp4 (output)")
    
    print("\n🚀 Usage with .ipynb files:")
    print("processor = VideoChunkProcessor(")
    print("    chunks_folder='chunks',")
    print("    modifications_json='modifications.json',")
    print("    output_video_path='final_video.mp4',")
    print("    filters_file='filters.ipynb',")
    print("    transitions_file='transitions_videos.ipynb'")
    print(")")
    print("processor.process_chunks()")
    
    print("\n🚀 Usage with .py files:")
    print("processor = VideoChunkProcessor(")
    print("    chunks_folder='chunks',")
    print("    modifications_json='modifications.json',")
    print("    output_video_path='final_video.mp4',")
    print("    filters_file='filters.py',")
    print("    transitions_file='transitions.py'")
    print(")")
    print("processor.process_chunks()")
    
    print("\n📝 For .ipynb support, install:")
    print("pip install nbformat nbconvert")


if __name__ == "__main__":
    main()