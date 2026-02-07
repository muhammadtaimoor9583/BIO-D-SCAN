import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType

class InsectDetector:
    def __init__(self, hef_path, labels):
        self.hef = HEF(hef_path)
        self.labels = labels
        self.target = VDevice()
        
        # Configure the Hailo device
        # NEW FIXED CODE
        self.configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, 
            interface=HailoStreamInterface.PCIe
        )
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        
        # Create input/output streams
        self.input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_height = self.input_vstream_info.shape[0]
        self.input_width = self.input_vstream_info.shape[1]

    def infer(self, image):
        # Resize image to model input size (e.g., 640x640)
        import cv2
        resized_img = cv2.resize(image, (self.input_width, self.input_height))
        # Add batch dimension
        input_data = np.expand_dims(resized_img, axis=0).astype(np.float32)

        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            # Send data
            infer_pipeline.input_vstream.send(input_data)
            # Receive result
            raw_output = infer_pipeline.output_vstream.recv()
            
        # NOTE: You usually need to decode raw_output here (NMS). 
        # If your HEF was compiled with NMS, the output is bounding boxes.
        # If not, you get raw grid scores and need a 'decode_yolo' function.
        # Assuming HEF has NMS baked in:
        return self.parse_hailo_output(raw_output)

    def parse_hailo_output(self, output):
        # Simplified parser - assumes output is [x1, y1, x2, y2, score, class_id]
        # You may need to adjust based on your specific HEF export settings
        detections = []
        for det in output['detections'][0]:
            if det[4] > 0.5: # Confidence threshold
                detections.append(det) 
        return detections