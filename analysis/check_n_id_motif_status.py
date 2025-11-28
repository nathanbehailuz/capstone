import pickle as pkl
import os
import numpy as np
import re
import sys

# Mock scipy to avoid import errors during unpickling if possible
# (This rarely works for unpickling complex types but worth a try if it's just looking for the module)
try:
    import scipy
except ImportError:
    import types
    sys.modules['scipy'] = types.ModuleType('scipy')
    sys.modules['scipy.sparse'] = types.ModuleType('scipy.sparse')

def check_motif_status_direct():
    print("Loading syn2 dataset directly from pickle...")
    
    # Path to pkl
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/pkls/syn2.pkl')
    
    try:
        with open(path, 'rb') as fin:
            # syn2 pkl structure: adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix
            data = pkl.load(fin)
            
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = data
        
        # Combine labels
        labels = y_train
        labels[val_mask] = y_val[val_mask]
        labels[test_mask] = y_test[test_mask]
        
        # Labels are likely one-hot, convert to indices
        if len(labels.shape) > 1:
            labels = np.argmax(labels, axis=1)
            
        unique_labels = np.unique(labels)
        print(f"Dataset loaded successfully.")
        print(f"Total nodes: {len(labels)}")
        print(f"Unique labels: {unique_labels}")
        
    except Exception as e:
        print(f"Failed to load pickle: {e}")
        print("Falling back to heuristic check based on typical syn2 structure.")
        # Fallback to heuristic
        # syn2 usually: 0-699 are base (class 0), 700-1399 are motifs (class 1-4)
        labels = np.zeros(1400, dtype=int)
        labels[700:] = 1 # Assume >700 is motif
        
    
    # Provided data snippet
    data_snippet = """
    n_id 400 len 15 p_loss=4.41 size_loss=0.16 sim_loss=1.88 r_p=0.40 sgs [0, 3, 1, -340, -277, -226, -332, 2, -132, -363, -309, -183, -252, -337, -243]
    n_id 405 len 15 p_loss=2.72 size_loss=0.16 sim_loss=1.53 r_p=0.50 sgs [0, 3, 894, 299, 300, 583, 464, 326, 314, 312, 305, 423, 558, 398, 327]
    n_id 410 len 15 p_loss=1.51 size_loss=0.15 sim_loss=1.29 r_p=0.50 sgs [0, 3, 1, 4, 682, 602, 683, 349, 553, 295, 459, 549, 423, 331, 471]
    n_id 415 len 15 p_loss=5.06 size_loss=0.14 sim_loss=2.13 r_p=0.70 sgs [0, 1, 3, 503, 514, 200, 245, 304, 556, -226, 130, 505, -370, -414, -130]
    n_id 420 len 12 p_loss=0.53 size_loss=0.12 sim_loss=0.42 r_p=0.30 sgs [0, 3, 4, -348, -142, -419, -247, -267, -189, -383, -366, -124]
    n_id 425 len 15 p_loss=1.29 size_loss=0.14 sim_loss=1.40 r_p=0.50 sgs [0, 3, 4, 339, 352, 301, 430, 363, 434, 692, 392, 180, -242, -158, 562]
    n_id 430 len 15 p_loss=3.35 size_loss=0.14 sim_loss=1.83 r_p=0.60 sgs [0, 876, -352, 877, 875, 453, 1, 451, 58, -380, 452, 540, -277, 436, 57]
    n_id 435 len 15 p_loss=4.45 size_loss=0.19 sim_loss=1.88 r_p=0.40 sgs [0, -354, -350, -426, -425, -414, -335, -412, -292, -328, -217, -138, -421, -394, -235]
    n_id 440 len 8 p_loss=1.23 size_loss=0.07 sim_loss=1.85 r_p=0.40 sgs [0, 4, -356, -212, -428, -401, -201, -383]
    n_id 445 len 15 p_loss=1.64 size_loss=0.16 sim_loss=1.10 r_p=0.50 sgs [0, 3, 2, 1, -358, -322, -390, -407, -166, -285, -361, -431, -416, 299, -320]
    n_id 450 len 15 p_loss=1.07 size_loss=0.14 sim_loss=0.68 r_p=0.50 sgs [0, 4, -360, -373, 3, -442, -384, -404, -40, -435, -400, -397, -338, -401, -425]
    n_id 455 len 10 p_loss=0.76 size_loss=0.09 sim_loss=0.91 r_p=0.50 sgs [0, -362, -395, 1, -339, -440, -189, -341, -400, -265]
    n_id 460 len 15 p_loss=1.84 size_loss=0.14 sim_loss=1.02 r_p=0.50 sgs [0, 3, -364, -376, -20, -19, 788, -290, -178, -195, -442, -332, -392, -271, -216]
    n_id 465 len 15 p_loss=1.58 size_loss=0.15 sim_loss=1.09 r_p=0.50 sgs [0, 760, -366, -179, -332, -362, 4, -420, 3, -323, -276, -432, -213, -258, -232]
    n_id 470 len 15 p_loss=5.37 size_loss=0.15 sim_loss=1.82 r_p=0.50 sgs [0, 4, 1, 829, 234, 408, -65, -61, -62, 235, 391, 292, 273, 405, 275]
    n_id 475 len 15 p_loss=0.80 size_loss=0.15 sim_loss=1.07 r_p=0.50 sgs [0, -370, -460, -399, -415, -230, -189, -463, -357, -472, -296, -210, 888, -403, -285]
    n_id 480 len 15 p_loss=2.58 size_loss=0.15 sim_loss=1.78 r_p=0.40 sgs [0, 4, -372, -414, 759, 755, -437, 837, -458, -479, -446, -465, -242, -425, -317]
    n_id 485 len 15 p_loss=1.00 size_loss=0.20 sim_loss=1.45 r_p=0.50 sgs [0, 1, 2, 4, 316, 258, -374, 243, 283, 3, 238, 410, 275, 381, 314]
    n_id 490 len 12 p_loss=0.26 size_loss=0.11 sim_loss=0.29 r_p=0.50 sgs [0, 4, -376, 3, -444, -442, -353, -485, -313, -449, -285, -365]
    n_id 495 len 15 p_loss=0.75 size_loss=0.16 sim_loss=0.66 r_p=0.50 sgs [0, -378, 4, -465, -145, -421, -141, -444, -364, 1, -326, -389, -456, -311, -310]
    n_id 500 len 15 p_loss=0.33 size_loss=0.15 sim_loss=0.79 r_p=0.60 sgs [0, 3, 2, 752, 753, 1, -453, -454, -368, -354, -422, -409, -435, -341, -496]
    n_id 505 len 15 p_loss=1.10 size_loss=0.17 sim_loss=0.74 r_p=0.50 sgs [0, 3, 2, -382, 1, -348, -500, -237, -331, -394, -504, -503, -445, -484, -317]
    n_id 510 len 15 p_loss=0.71 size_loss=0.16 sim_loss=0.40 r_p=0.40 sgs [0, 3, 4, -197, 737, 2, -384, -339, 507, 1, -501, -200, -453, -495, -473]
    n_id 515 len 15 p_loss=0.64 size_loss=0.16 sim_loss=0.71 r_p=0.80 sgs [0, 1, 2, 3, 4, 278, 197, 221, 218, -332, 369, -375, -495, 515, -492]
    n_id 520 len 15 p_loss=2.85 size_loss=0.15 sim_loss=1.63 r_p=0.40 sgs [0, 4, -388, -493, -481, -421, -502, -294, -455, -418, -486, -505, -463, -175, -392]
    n_id 525 len 15 p_loss=2.74 size_loss=0.16 sim_loss=2.05 r_p=0.60 sgs [0, 3, 1, -390, -394, 351, -518, -520, -513, -321, -507, -523, -453, -246, 120]
    n_id 530 len 15 p_loss=1.29 size_loss=0.16 sim_loss=0.81 r_p=0.50 sgs [0, -392, -425, -266, -312, -265, -380, -465, -517, -439, -527, -263, -249, -448, -273]
    n_id 535 len 15 p_loss=2.47 size_loss=0.14 sim_loss=1.72 r_p=0.50 sgs [0, 3, 4, -394, -531, -455, -522, -478, -244, -257, -459, -524, -302, -450, -482]
    n_id 540 len 15 p_loss=0.84 size_loss=0.15 sim_loss=1.20 r_p=0.50 sgs [0, 3, -396, 1, -376, -528, -515, -314, -474, -379, -433, -511, -487, -263, -516]
    n_id 545 len 15 p_loss=2.60 size_loss=0.14 sim_loss=1.69 r_p=0.50 sgs [0, 1, 3, 660, -398, -358, 568, -496, -537, 567, -512, -474, -283, 664, -414]
    n_id 550 len 10 p_loss=0.31 size_loss=0.11 sim_loss=0.92 r_p=0.30 sgs [0, 3, 4, 1, 2, -400, -361, -511, -389, -461]
    n_id 555 len 15 p_loss=0.60 size_loss=0.15 sim_loss=0.79 r_p=0.60 sgs [0, 4, -402, -498, -554, -284, -550, -532, -358, -292, -384, -419, -479, -332, -452]
    n_id 560 len 15 p_loss=0.23 size_loss=0.15 sim_loss=0.78 r_p=0.40 sgs [0, 4, -404, 1, -522, -400, -342, -390, -455, -85, -436, -559, -466, -476, -425]
    n_id 565 len 11 p_loss=0.91 size_loss=0.10 sim_loss=0.81 r_p=0.50 sgs [0, 1, 3, 755, 28, 27, 732, 756, 25, -406, -391]
    n_id 570 len 15 p_loss=5.04 size_loss=0.14 sim_loss=1.77 r_p=0.50 sgs [0, 1, 2, -408, -519, -550, -553, -421, -482, -436, -538, -530, -542, -272, -395]
    n_id 575 len 15 p_loss=0.29 size_loss=0.17 sim_loss=0.59 r_p=0.50 sgs [0, 4, 3, 2, 1, -410, -496, -491, -310, -557, -529, -390, -492, -521, -549]
    n_id 580 len 15 p_loss=0.40 size_loss=0.16 sim_loss=0.83 r_p=0.30 sgs [0, 3, 4, -412, -365, -571, -265, -394, -380, -494, -451, -495, -310, -416, -402]
    n_id 585 len 15 p_loss=3.86 size_loss=0.14 sim_loss=2.11 r_p=0.40 sgs [0, 3, 276, -414, -459, -390, -517, 123, -533, 387, 153, 414, 216, 279, -505]
    n_id 590 len 15 p_loss=0.65 size_loss=0.15 sim_loss=1.32 r_p=0.50 sgs [0, 3, 730, -416, 1, -467, -24, -23, -553, 733, -518, -384, 2, -419, -464]
    n_id 595 len 15 p_loss=2.88 size_loss=0.15 sim_loss=2.17 r_p=0.40 sgs [0, -418, -296, -414, 227, -579, -520, -583, 311, 230, -369, -318, -593, -508, -502]
    n_id 600 len 15 p_loss=0.33 size_loss=0.15 sim_loss=0.98 r_p=0.60 sgs [0, 4, 1, 634, 554, 631, 632, 551, 30, 34, 3, 228, 185, 247, 633]
    n_id 605 len 15 p_loss=6.48 size_loss=0.16 sim_loss=2.15 r_p=0.30 sgs [0, 172, 339, 158, 184, 389, 215, 358, 354, 373, 318, 260, 352, 107, 114]
    n_id 610 len 15 p_loss=5.18 size_loss=0.17 sim_loss=1.99 r_p=0.50 sgs [0, 3, -424, 2, -581, -605, -328, -474, -537, -602, -606, -354, -509, -569, -535]
    n_id 615 len 15 p_loss=1.32 size_loss=0.14 sim_loss=1.15 r_p=0.40 sgs [0, 303, -197, 1, -198, 2, 285, 348, -181, -184, 185, 108, 95, 209, -200]
    n_id 620 len 15 p_loss=0.06 size_loss=0.14 sim_loss=0.92 r_p=0.40 sgs [0, 4, 3, -428, -596, -601, -619, -280, -357, -404, -351, -586, -370, -457, -443]
    n_id 625 len 15 p_loss=5.87 size_loss=0.16 sim_loss=2.05 r_p=0.70 sgs [0, 4, -430, -557, -533, -588, 1, -608, -391, -421, -331, -501, -594, -610, -604]
    n_id 630 len 15 p_loss=4.31 size_loss=0.16 sim_loss=1.93 r_p=0.60 sgs [0, 4, 217, 601, 73, 114, 152, 198, -575, 87, 81, -379, 136, 83, 235]
    n_id 635 len 10 p_loss=1.01 size_loss=0.10 sim_loss=1.36 r_p=0.60 sgs [0, 1, 2, 3, 72, 220, 76, 71, 282, 332]
    n_id 640 len 15 p_loss=2.26 size_loss=0.15 sim_loss=1.10 r_p=0.40 sgs [0, 4, 3, -436, -385, -635, -549, -592, -418, -504, -460, -588, 461, -593, -619]
    n_id 645 len 15 p_loss=0.27 size_loss=0.16 sim_loss=0.44 r_p=0.30 sgs [0, 3, 4, -573, 2, 1, -225, 422, -642, -598, -579, -222, -634, -492, -399]
    """
    
    print("\nProcessing node IDs from data snippet...")
    print(f"{'Node ID':<10} | {'Label':<10} | {'Type':<15} | {'Motif Status':<15}")
    print("-" * 60)
    
    lines = data_snippet.strip().split('\n')
    motif_count = 0
    base_count = 0
    total = 0
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Parse n_id using regex
        match = re.search(r'n_id\s+(\d+)', line)
        if match:
            n_id = int(match.group(1))
            total += 1
            
            if n_id >= len(labels):
                print(f"{n_id:<10} | {'N/A':<10} | {'Out of range':<15} | {'Unknown':<15}")
                continue
                
            label = labels[n_id]
            
            # Determine type based on label and index for syn2
            # In syn2 (BA-Community):
            # - Nodes 0-699 are Base Graph (assigned to communities 0, 1, 2, 3)
            # - Nodes 700-1399 are Motif Nodes (assigned to roles 4, 5, 6, 7)
            
            is_motif = False
            if n_id >= 700:
                is_motif = True
            elif label >= 4:
                is_motif = True
                
            if is_motif:
                node_type = f"Motif ({label})"
                status = "MOTIF"
                motif_count += 1
            else:
                node_type = f"Base ({label})"
                status = "Base Graph"
                base_count += 1
                
            print(f"{n_id:<10} | {label:<10} | {node_type:<15} | {status:<15}")
            
    print("-" * 60)
    print(f"Total processed: {total}")
    print(f"Motif nodes: {motif_count}")
    print(f"Base nodes: {base_count}")
    print(f"Ratio (Motif/Total): {motif_count/total:.2f}")

if __name__ == "__main__":
    check_motif_status_direct()
