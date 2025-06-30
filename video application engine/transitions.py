import cv2
import numpy as np

def crossfade_transition(frames1, frames2, transition_duration=30):
    """Smooth crossfade between two chunks"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    # Pre-transition frames (all except the last transition_frames)
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    
    # Transition frames
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    
    # Post-transition frames
    post_transition = frames2[transition_frames:]
    
    result_frames = []
    
    # Add pre-transition frames
    result_frames.extend(pre_transition)
    
    # Create crossfade frames
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        blended = cv2.addWeighted(trans1[i], 1 - alpha, trans2[i], alpha, 0)
        result_frames.append(blended)
    
    # Add post-transition frames
    result_frames.extend(post_transition)
    
    return result_frames

def slide_left_transition(frames1, frames2, transition_duration=30):
    """Slide left transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Slide transition
    for i in range(min(len(trans1), len(trans2))):
        dx = int(width * i / len(trans1))
        slide = np.zeros_like(trans1[i])
        slide[:, :width - dx] = trans1[i][:, dx:]
        slide[:, width - dx:] = trans2[i][:, :dx]
        result_frames.append(slide)
    
    result_frames.extend(post_transition)
    return result_frames

def wipe_right_transition(frames1, frames2, transition_duration=30):
    """Wipe right transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Wipe transition
    for i in range(min(len(trans1), len(trans2))):
        wipe = trans1[i].copy()
        x = int(width * i / len(trans1))
        wipe[:, :x] = trans2[i][:, :x]
        result_frames.append(wipe)
    
    result_frames.extend(post_transition)
    return result_frames

def zoom_out_transition(frames1, frames2, transition_duration=30):
    """Zoom out transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Zoom out transition
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        scale = 1.5 - 0.5 * alpha  # Zoom from 1.5x to 1x
        
        zoomed = cv2.resize(trans2[i], None, fx=scale, fy=scale)
        y = max(0, (zoomed.shape[0] - height) // 2)
        x = max(0, (zoomed.shape[1] - width) // 2)
        
        # Ensure we don't go out of bounds
        y_end = min(zoomed.shape[0], y + height)
        x_end = min(zoomed.shape[1], x + width)
        
        zoomed_crop = zoomed[y:y_end, x:x_end]
        
        # If cropped image is smaller than target, pad it
        if zoomed_crop.shape[:2] != (height, width):
            zoomed_crop = cv2.resize(zoomed_crop, (width, height))
        
        blended = cv2.addWeighted(trans1[i], 1 - alpha, zoomed_crop, alpha, 0)
        result_frames.append(blended)
    
    result_frames.extend(post_transition)
    return result_frames

def fade_to_black_transition(frames1, frames2, transition_duration=30):
    """Fade to black transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    half_transition = transition_frames // 2
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    fadeout_frames = frames1[-transition_frames:-half_transition] if len(frames1) >= transition_frames else frames1
    fadein_frames = frames2[:half_transition]
    post_transition = frames2[half_transition:]
    
    result_frames = []
    result_frames.extend(pre_transition)
    
    # Fade to black
    for i, frame in enumerate(fadeout_frames):
        alpha = 1 - (i / len(fadeout_frames))
        black = np.zeros_like(frame)
        blended = cv2.addWeighted(frame, alpha, black, 1 - alpha, 0)
        result_frames.append(blended)
    
    # Fade in from black
    for i, frame in enumerate(fadein_frames):
        alpha = i / len(fadein_frames)
        black = np.zeros_like(frame)
        blended = cv2.addWeighted(black, 1 - alpha, frame, alpha, 0)
        result_frames.append(blended)
    
    result_frames.extend(post_transition)
    return result_frames

def diagonal_wipe_transition(frames1, frames2, transition_duration=30):
    """Diagonal wipe transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Diagonal wipe
    for i in range(min(len(trans1), len(trans2))):
        limit = int((i / len(trans1)) * (width + height))
        mask = trans1[i].copy()
        
        for y in range(height):
            for x in range(width):
                if x + y < limit:
                    mask[y, x] = trans2[i][y, x]
        
        result_frames.append(mask)
    
    result_frames.extend(post_transition)
    return result_frames

def split_horizontal_transition(frames1, frames2, transition_duration=30):
    """Split horizontal transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Split horizontal transition
    for i in range(min(len(trans1), len(trans2))):
        offset = int((i / len(trans1)) * (height // 2))
        frame = trans1[i].copy()
        
        if offset > 0:
            frame[:offset, :] = trans2[i][:offset, :]
            frame[height - offset:, :] = trans2[i][height - offset:, :]
        
        result_frames.append(frame)
    
    result_frames.extend(post_transition)
    return result_frames

def circle_reveal_transition(frames1, frames2, transition_duration=30):
    """Circle reveal transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    center = (width // 2, height // 2)
    max_radius = int(np.hypot(width, height) / 2)
    
    result_frames = []
    result_frames.extend(pre_transition)
    
    # Circle reveal effect
    for i in range(min(len(trans1), len(trans2))):
        radius = int((i / len(trans1)) * max_radius)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        mask_3ch = cv2.merge([mask, mask, mask])
        blended = np.where(mask_3ch == 255, trans2[i], trans1[i])
        result_frames.append(blended.astype(np.uint8))
    
    result_frames.extend(post_transition)
    return result_frames

def pixel_dissolve_transition(frames1, frames2, transition_duration=30):
    """Pixel dissolve transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    total_pixels = width * height
    indices = np.arange(total_pixels)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    result_frames = []
    result_frames.extend(pre_transition)
    
    # Pixel dissolve effect
    mask = np.zeros((height, width), dtype=np.uint8)
    pixels_per_frame = max(1, total_pixels // transition_frames)
    
    for i in range(min(len(trans1), len(trans2))):
        current_indices = indices[:(i + 1) * pixels_per_frame]
        current_indices = current_indices[current_indices < total_pixels]
        mask.flat[current_indices] = 255
        mask_rgb = cv2.merge([mask, mask, mask])
        blended = np.where(mask_rgb == 255, trans2[i], trans1[i])
        result_frames.append(blended.astype(np.uint8))
    
    result_frames.extend(post_transition)
    return result_frames

def wave_slide_transition(frames1, frames2, transition_duration=30):
    """Wave slide transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Wave-slide transition
    for step in range(min(len(trans1), len(trans2))):
        alpha = step / len(trans1)
        shift = int(alpha * width)
        wave_frame = np.zeros_like(trans1[step])
        
        for y in range(height):
            wave_offset = int(20 * np.sin(2 * np.pi * y / 100 + alpha * 5))
            src_x = min(width, max(0, shift + wave_offset))
            
            if src_x < width:
                wave_frame[y, :width - src_x] = trans1[step][y, src_x:]
                wave_frame[y, width - src_x:] = trans2[step][y, :src_x]
            else:
                wave_frame[y] = trans2[step][y]
        
        result_frames.append(wave_frame)
    
    result_frames.extend(post_transition)
    return result_frames

def zoom_blur_transition(frames1, frames2, transition_duration=30):
    """Zoom blur transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        scale = 1 + 0.5 * alpha
        blur_amount = int((1 - alpha) * 21)
        if blur_amount % 2 == 0:
            blur_amount += 1
        
        zoomed = cv2.resize(trans1[i], None, fx=scale, fy=scale)
        y, x = max(0, (zoomed.shape[0] - height) // 2), max(0, (zoomed.shape[1] - width) // 2)
        
        # Ensure we don't go out of bounds
        y_end = min(zoomed.shape[0], y + height)
        x_end = min(zoomed.shape[1], x + width)
        cropped = zoomed[y:y_end, x:x_end]
        
        # Resize if needed
        if cropped.shape[:2] != (height, width):
            cropped = cv2.resize(cropped, (width, height))
        
        if blur_amount > 1:
            cropped = cv2.GaussianBlur(cropped, (blur_amount, blur_amount), 0)
        
        blended = cv2.addWeighted(cropped, 1 - alpha, trans2[i], alpha, 0)
        result_frames.append(blended)
    
    result_frames.extend(post_transition)
    return result_frames

def vertical_uncover_transition(frames1, frames2, transition_duration=30):
    """Vertical uncover transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    for i in range(min(len(trans1), len(trans2))):
        cut = int((i / len(trans1)) * height)
        combined = trans1[i].copy()
        if cut > 0:
            combined[:cut, :] = trans2[i][:cut, :]
        result_frames.append(combined)
    
    result_frames.extend(post_transition)
    return result_frames

def radial_wipe_transition(frames1, frames2, transition_duration=30):
    """Radial wipe transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    center = (width // 2, height // 2)
    radius = int(np.hypot(width, height))
    
    result_frames = []
    result_frames.extend(pre_transition)
    
    for i in range(min(len(trans1), len(trans2))):
        angle = int(360 * i / len(trans1))
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, center, (radius, radius), 0, 0, angle, 255, -1)
        mask_rgb = cv2.merge([mask, mask, mask])
        blended = np.where(mask_rgb == 255, trans2[i], trans1[i])
        result_frames.append(blended.astype(np.uint8))
    
    result_frames.extend(post_transition)
    return result_frames

def checkerboard_transition(frames1, frames2, transition_duration=30):
    """Checkerboard transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    squares = 10
    block_h = height // squares
    block_w = width // squares
    total_blocks = squares * squares
    block_indices = [(i, j) for i in range(squares) for j in range(squares)]
    np.random.seed(0)
    np.random.shuffle(block_indices)
    
    result_frames = []
    result_frames.extend(pre_transition)
    
    for i in range(min(len(trans1), len(trans2))):
        n_blocks = int((i / len(trans1)) * total_blocks)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for b in block_indices[:n_blocks]:
            y, x = b
            y1, y2 = y * block_h, min((y + 1) * block_h, height)
            x1, x2 = x * block_w, min((x + 1) * block_w, width)
            mask[y1:y2, x1:x2] = 255
        
        mask_rgb = cv2.merge([mask, mask, mask])
        blended = np.where(mask_rgb == 255, trans2[i], trans1[i])
        result_frames.append(blended.astype(np.uint8))
    
    result_frames.extend(post_transition)
    return result_frames

def curtain_open_transition(frames1, frames2, transition_duration=30):
    """Curtain open transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Curtain open effect
    for step in range(min(len(trans1), len(trans2))):
        offset = int((step / len(trans1)) * (width // 2))
        combined = trans1[step].copy()
        
        if offset > 0:
            combined[:, :offset] = trans2[step][:, :offset]
            combined[:, width - offset:] = trans2[step][:, width - offset:]
        
        result_frames.append(combined)
    
    result_frames.extend(post_transition)
    return result_frames

def iris_box_transition(frames1, frames2, transition_duration=30):
    """Iris box transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    height, width = frames1[0].shape[:2]
    center_x, center_y = width // 2, height // 2
    
    result_frames = []
    result_frames.extend(pre_transition)
    
    # Iris box effect
    for i in range(min(len(trans1), len(trans2))):
        box_width = int((i / len(trans1)) * width)
        box_height = int((i / len(trans1)) * height)
        x1, x2 = max(0, center_x - box_width // 2), min(width, center_x + box_width // 2)
        y1, y2 = max(0, center_y - box_height // 2), min(height, center_y + box_height // 2)
        
        mask = trans1[i].copy()
        mask[y1:y2, x1:x2] = trans2[i][y1:y2, x1:x2]
        result_frames.append(mask)
    
    result_frames.extend(post_transition)
    return result_frames

def glitch_transition(frames1, frames2, transition_duration=30):
    """Glitch transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    result_frames = []
    np.random.seed(0)
    
    result_frames.extend(pre_transition)
    
    # Glitch transition effect
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        mask = np.random.choice([0, 1], size=(h,), p=[1 - alpha, alpha])
        
        frame = trans1[i].copy()
        for y in range(h):
            if mask[y]:
                glitch_shift = np.random.randint(-20, 20)
                x1 = max(0, glitch_shift)
                x2 = min(w, w + glitch_shift)
                if x2 - x1 > 0:
                    frame[y, :x2 - x1] = trans2[i][y, x1:x2]
        
        result_frames.append(frame)
    
    result_frames.extend(post_transition)
    return result_frames

def rgb_split_transition(frames1, frames2, transition_duration=30):
    """RGB split transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    result_frames = []
    result_frames.extend(pre_transition)
    
    # RGB Split transition effect
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        shift = int(10 * (1 - alpha))
        
        b1, g1, r1 = cv2.split(trans1[i])
        
        r1_shifted = np.roll(r1, shift, axis=1)     # Horizontal shift (red)
        g1_shifted = np.roll(g1, -shift, axis=0)    # Vertical upward shift (green)
        b1_shifted = np.roll(b1, shift, axis=0)     # Vertical downward shift (blue)
        
        rgb_glitch = cv2.merge([b1_shifted, g1_shifted, r1_shifted])
        blended = cv2.addWeighted(rgb_glitch, 1 - alpha, trans2[i], alpha, 0)
        result_frames.append(blended)
    
    result_frames.extend(post_transition)
    return result_frames

def door_open_transition(frames1, frames2, transition_duration=30):
    """Door open transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    half_w = w // 2
    
    result_frames = []
    result_frames.extend(pre_transition)
    
    # Door opening transition
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        panel_width = int(half_w * (1 - alpha))
        frame = trans2[i].copy()
        
        if panel_width > 0:
            frame[:, :panel_width] = trans1[i][:, :panel_width]
            frame[:, w - panel_width:] = trans1[i][:, w - panel_width:]
        
        result_frames.append(frame)
    
    result_frames.extend(post_transition)
    return result_frames

def horizontal_ripple_transition(frames1, frames2, transition_duration=30):
    """Horizontal ripple transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Horizontal ripple transition
    for step in range(min(len(trans1), len(trans2))):
        alpha = step / len(trans1)
        ripple_frame = np.zeros_like(trans1[step])
        
        for y in range(h):
            offset = int(10 * np.sin(2 * np.pi * y / 30 + alpha * 5))
            if offset > 0:
                ripple_frame[y, offset:] = trans1[step][y, :-offset]
            elif offset < 0:
                ripple_frame[y, :offset] = trans1[step][y, -offset:]
            else:
                ripple_frame[y] = trans1[step][y]
        
        blended = cv2.addWeighted(ripple_frame, 1 - alpha, trans2[step], alpha, 0)
        result_frames.append(blended)
    
    result_frames.extend(post_transition)
    return result_frames

def swirl_rotation_transition(frames1, frames2, transition_duration=30):
    """Swirl rotation transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    center = (w // 2, h // 2)
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Swirl rotation transition
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        angle = (1 - alpha) * 180
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(trans1[i], M, (w, h))
        blended = cv2.addWeighted(rotated, 1 - alpha, trans2[i], alpha, 0)
        result_frames.append(blended)
    
    result_frames.extend(post_transition)
    return result_frames

def tile_collapse_transition(frames1, frames2, transition_duration=30):
    """Tile collapse transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    tiles_x, tiles_y = 8, 6
    tile_w, tile_h = w // tiles_x, h // tiles_y
    total_tiles = tiles_x * tiles_y
    tile_indices = [(i, j) for i in range(tiles_y) for j in range(tiles_x)]
    np.random.seed(42)
    np.random.shuffle(tile_indices)
    
    result_frames = []
    result_frames.extend(pre_transition)
    
    # Tile collapse transition frames
    for step in range(min(len(trans1), len(trans2))):
        current_tiles = int((step / len(trans1)) * total_tiles)
        frame = trans1[step].copy()
        
        for i, j in tile_indices[:current_tiles]:
            y1, y2 = i * tile_h, min((i + 1) * tile_h, h)
            x1, x2 = j * tile_w, min((j + 1) * tile_w, w)
            frame[y1:y2, x1:x2] = trans2[step][y1:y2, x1:x2]
        
        result_frames.append(frame)
    
    result_frames.extend(post_transition)
    return result_frames

def tv_static_transition(frames1, frames2, transition_duration=30):
    """TV static transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Static glitch transition frames
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        noise = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        noise_img = cv2.merge([noise, noise, noise])
        blended = cv2.addWeighted(trans1[i], 1 - alpha, trans2[i], alpha, 0.5)
        static_blend = cv2.addWeighted(blended, 0.7, noise_img, 0.3, 0)
        result_frames.append(static_blend)
    
    result_frames.extend(post_transition)
    return result_frames

def cross_zoom_transition(frames1, frames2, transition_duration=30):
    """Cross zoom transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Cross zoom transition frames
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        zoom_frame = np.zeros_like(trans1[i], dtype=np.float32)
        
        for k in range(5):  # simulate zoom blur with multiple scales
            scale = 1 + (1 - alpha) * 0.2 * k
            resized = cv2.resize(trans1[i], None, fx=scale, fy=scale)
            y_offset = max((resized.shape[0] - h) // 2, 0)
            x_offset = max((resized.shape[1] - w) // 2, 0)
            
            # Ensure we don't go out of bounds
            y_end = min(resized.shape[0], y_offset + h)
            x_end = min(resized.shape[1], x_offset + w)
            cropped = resized[y_offset:y_end, x_offset:x_end]
            
            # Resize if needed
            if cropped.shape[:2] != (h, w):
                cropped = cv2.resize(cropped, (w, h))
            
            zoom_frame += cropped.astype(np.float32)
        
        zoom_frame /= 5.0
        blended = cv2.addWeighted(zoom_frame.astype(np.uint8), 1 - alpha, trans2[i], alpha, 0)
        result_frames.append(blended)
    
    result_frames.extend(post_transition)
    return result_frames

def page_turn_transition(frames1, frames2, transition_duration=30):
    """Page turn transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Page turn effect
    for i in range(min(len(trans1), len(trans2))):
        offset = int((i / len(trans1)) * w)
        frame = trans1[i].copy()
        if offset > 0:
            frame[:, w - offset:] = trans2[i][:, w - offset:]
        result_frames.append(frame)
    
    result_frames.extend(post_transition)
    return result_frames

def wave_distortion_transition(frames1, frames2, transition_duration=30):
    """Wave distortion transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    x = np.arange(w)
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Wave distortion effect
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        displacement = (np.sin(2 * np.pi * x / 50 + i * 0.3) * 10).astype(np.int32)
        frame = np.zeros_like(trans1[i])
        
        for y in range(h):
            row = np.clip(np.arange(w) + displacement, 0, w - 1)
            frame[y] = (1 - alpha) * trans1[i][y, row] + alpha * trans2[i][y, row]
        
        result_frames.append(frame.astype(np.uint8))
    
    result_frames.extend(post_transition)
    return result_frames

def hexagon_wipe_transition(frames1, frames2, transition_duration=30):
    """Hexagon wipe transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    hex_size = 50
    centers = [(x, y) for y in range(0, h, hex_size) for x in range(0, w, int(hex_size * 0.866))]
    
    np.random.seed(0)
    np.random.shuffle(centers)
    
    result_frames = []
    result_frames.extend(pre_transition)
    
    # Hexagon wipe transition
    for i in range(min(len(trans1), len(trans2))):
        threshold = int((i / len(trans1)) * len(centers))
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for cx, cy in centers[:threshold]:
            cv2.circle(mask, (cx, cy), hex_size // 2, 255, -1)
        
        mask_3c = cv2.merge([mask] * 3)
        blended = np.where(mask_3c == 255, trans2[i], trans1[i])
        result_frames.append(blended)
    
    result_frames.extend(post_transition)
    return result_frames

def glitch_distortion_transition(frames1, frames2, transition_duration=30):
    """Glitch distortion transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Glitch distortion transition
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        glitch_frame = trans1[i].copy()
        
        # Horizontal line jitter
        for j in range(0, h, 10):
            offset = np.random.randint(-20, 20)
            if j + 5 <= h:
                glitch_frame[j:j+5] = np.roll(glitch_frame[j:j+5], offset, axis=1)
        
        # RGB split distortion
        b, g, r = cv2.split(glitch_frame)
        b = np.roll(b, np.random.randint(-5, 5), axis=1)
        r = np.roll(r, np.random.randint(-5, 5), axis=1)
        glitch_frame = cv2.merge([b, g, r])
        
        # Jitter frame2 as target blend
        jittered = np.roll(trans2[i], np.random.randint(-5, 5), axis=1) if np.random.rand() > 0.5 else trans2[i]
        blended = cv2.addWeighted(glitch_frame, 1 - alpha, jittered, alpha, 0)
        
        result_frames.append(blended)
    
    result_frames.extend(post_transition)
    return result_frames

def generate_noise_mask(shape, scale=10):
    """Generate noise mask for liquid ink splash effect"""
    noise = np.random.rand(*shape[:2])
    blur = cv2.GaussianBlur(noise, (scale | 1, scale | 1), 0)
    normalized = cv2.normalize(blur, None, 0, 1, cv2.NORM_MINMAX)
    return normalized

def liquid_ink_splash_transition(frames1, frames2, transition_duration=30):
    """Liquid ink splash transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # Liquid ink splash transition
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        noise_mask = generate_noise_mask((h, w), scale=35)
        binary_mask = (noise_mask < alpha).astype(np.uint8)
        binary_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (21, 21), 0)
        binary_mask = np.clip(binary_mask, 0, 1)
        mask_3c = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
        frame = (trans1[i] * (1 - mask_3c) + trans2[i] * mask_3c).astype(np.uint8)
        result_frames.append(frame)
    
    result_frames.extend(post_transition)
    return result_frames

def flip_3d_perspective_transition(frames1, frames2, transition_duration=30):
    """3D perspective flip transition effect"""
    transition_frames = min(transition_duration, len(frames1), len(frames2))
    
    pre_transition = frames1[:-transition_frames] if len(frames1) > transition_frames else []
    trans1 = frames1[-transition_frames:]
    trans2 = frames2[:transition_frames]
    post_transition = frames2[transition_frames:]
    
    h, w = frames1[0].shape[:2]
    result_frames = []
    
    result_frames.extend(pre_transition)
    
    # 3D Perspective Flip
    for i in range(min(len(trans1), len(trans2))):
        alpha = i / len(trans1)
        angle = alpha * np.pi
        scale = abs(np.cos(angle))
        shrink = int(w * scale)
        f = trans1[i] if i <= len(trans1) // 2 else trans2[i]
        
        if shrink > 0:
            resized = cv2.resize(f, (shrink, h))
        else:
            resized = np.zeros((h, 1, 3), dtype=np.uint8)
        
        padding = (w - resized.shape[1]) // 2
        frame = np.zeros_like(f)
        
        if padding >= 0 and padding + resized.shape[1] <= w:
            frame[:, padding:padding + resized.shape[1]] = resized
        
        result_frames.append(frame)
    
    result_frames.extend(post_transition)
    return result_frames