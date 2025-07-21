import os
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

def load_structure_from_path(data_path):
    """Load PyTorch Geometric data directly from a file path."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No data file found at the specified path: {data_path}")
    
    print(f"Loading structure from:\n{os.path.normpath(data_path)}")
    return torch.load(data_path, weights_only=False)

class DiagnosticsDashboard(tk.Toplevel):
    """A GUI dashboard with a robust layout and intuitive scrolling."""
    def __init__(self, parent, structure_graph, file_name):
        super().__init__(parent)
        self.title(f"Diagnostics: {file_name}")
        self.geometry("1200x750") # A more standard, wider default size
        self.structure_graph = structure_graph
        
        main_frame = ttk.Frame(self)
        main_frame.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        
        # Use a frame specifically for the scrollable content
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Bind self.scrollable_frame to the canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel scrolling to the canvas and the inner frame
        for widget in [self.canvas, self.scrollable_frame]:
            widget.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.all_diagnostics_content = []

        # Use grid for the scrollable_frame to allow proportional expansion of sections
        self.scrollable_frame.grid_columnconfigure(0, weight=1) # Ensure columns expand horizontally

        row_idx = 0
        
        # Copy All Button
        copy_all_frame = ttk.Frame(self.scrollable_frame, padding=(10, 10))
        copy_all_frame.grid(row=row_idx, column=0, sticky='ew')
        self.create_copy_all_button_content(copy_all_frame)
        self.scrollable_frame.grid_rowconfigure(row_idx, weight=0) # Button row doesn't need to expand
        row_idx += 1

        # Properties Section
        properties_frame = self.create_properties_section_frame(self.scrollable_frame)
        properties_frame.grid(row=row_idx, column=0, sticky='nsew', padx=10, pady=5)
        self.all_diagnostics_content.append(self.populate_properties_section(properties_frame))
        self.scrollable_frame.grid_rowconfigure(row_idx, weight=1) # Allow properties section to expand
        row_idx += 1

        # Node Features Section
        node_features_frame = self.create_table_section_frame(self.scrollable_frame, "Node Features")
        node_features_frame.grid(row=row_idx, column=0, sticky='nsew', padx=10, pady=5)
        self.all_diagnostics_content.append(self.populate_node_features_section(node_features_frame))
        self.scrollable_frame.grid_rowconfigure(row_idx, weight=1) # Allow table section to expand
        row_idx += 1

        # Edge Features Section
        edge_features_frame = self.create_table_section_frame(self.scrollable_frame, "Edge Features")
        edge_features_frame.grid(row=row_idx, column=0, sticky='nsew', padx=10, pady=5)
        self.all_diagnostics_content.append(self.populate_edge_features_section(edge_features_frame))
        self.scrollable_frame.grid_rowconfigure(row_idx, weight=1) # Allow table section to expand
        row_idx += 1
        
        # Edge Index Section
        edge_index_frame = self.create_table_section_frame(self.scrollable_frame, "Edge Index")
        edge_index_frame.grid(row=row_idx, column=0, sticky='nsew', padx=10, pady=5)
        self.all_diagnostics_content.append(self.populate_edge_index_section(edge_index_frame))
        self.scrollable_frame.grid_rowconfigure(row_idx, weight=1) # Allow table section to expand
        row_idx += 1

        # Targets Section
        targets_frame = self.create_table_section_frame(self.scrollable_frame, "Target Values (y)")
        targets_frame.grid(row=row_idx, column=0, sticky='nsew', padx=10, pady=5)
        self.all_diagnostics_content.append(self.populate_targets_section(targets_frame))
        self.scrollable_frame.grid_rowconfigure(row_idx, weight=1) # Allow table section to expand
        row_idx += 1

        self.finalize_setup()

    def _on_mousewheel(self, event):
        # Determine the widget under the cursor
        widget = self.winfo_containing(event.x_root, event.y_root)
        
        # Traverse up the widget hierarchy to find a Treeview or a Canvas (for table scrolling)
        target_scroll_widget = None
        current_widget = widget
        while current_widget is not None:
            if isinstance(current_widget, ttk.Treeview):
                # If a Treeview is found, this is the most specific scrollable widget
                target_scroll_widget = current_widget
                break
            # Check if the current widget is a Canvas that is explicitly designed for horizontal table scrolling
            if isinstance(current_widget, tk.Canvas) and hasattr(current_widget, '_is_table_canvas'):
                target_scroll_widget = current_widget
                break
            current_widget = current_widget.master
            if current_widget is None: # Break to prevent infinite loop
                break

        if target_scroll_widget:
            if isinstance(target_scroll_widget, ttk.Treeview):
                # For Treeviews, only vertical scrolling (or Shift+horizontal) is direct
                if event.state & 0x1: # Check for Shift key (0x1 is ShiftMask)
                    target_scroll_widget.xview_scroll(int(-1*(event.delta/120)), "units")
                else:
                    target_scroll_widget.yview_scroll(int(-1*(event.delta/120)), "units")
            elif isinstance(target_scroll_widget, tk.Canvas):
                # For the dedicated table canvas, handle horizontal scrolling
                target_scroll_widget.xview_scroll(int(-1*(event.delta/120)), "units")
            return
        
        # Otherwise, scroll the main canvas vertically
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_copy_all_button_content(self, parent_frame):
        # This function now just adds the button to the provided frame
        copy_all_btn = ttk.Button(parent_frame, text="Copy All Diagnostics to Clipboard", style="Accent.TButton")
        copy_all_btn.pack(anchor='w')
        self.copy_all_button = copy_all_btn

    def finalize_setup(self):
        valid_content = [content for content in self.all_diagnostics_content if content]
        full_content = "\n\n".join(valid_content)
        self.copy_all_button.config(command=lambda: self.copy_to_clipboard(full_content, self.copy_all_button))

    def copy_to_clipboard(self, content, button):
        try:
            self.clipboard_clear()
            self.clipboard_append(content)
            original_text = button.cget("text")
            button.config(text="Copied!")
            self.after(1500, lambda: button.config(text=original_text))
        except tk.TclError:
            messagebox.showerror("Clipboard Error", "The selected data is too large to be copied to the clipboard.", parent=self)

    def create_properties_section_frame(self, parent):
        # This function now only creates and returns the LabelFrame
        return ttk.LabelFrame(parent, text=" Graph Properties ", padding=(10, 5))

    def populate_properties_section(self, frame):
        # This function populates the given frame and returns content for copying
        properties = {
            "Number of nodes": self.structure_graph.num_nodes,
            "Number of edges": self.structure_graph.num_edges,
            "Is directed": self.structure_graph.is_directed(),
            "Has isolated nodes": self.structure_graph.has_isolated_nodes(),
            "Has self-loops": self.structure_graph.has_self_loops(),
            "Graph keys": list(self.structure_graph.keys())
        }
        content_to_copy = "--- Graph Properties ---\n"
        for i, (key, value) in enumerate(properties.items()):
            ttk.Label(frame, text=f"{key}:", font=("Segoe UI", 10, "bold")).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            ttk.Label(frame, text=str(value), wraplength=500).grid(row=i, column=1, sticky='w', padx=5, pady=2)
            content_to_copy += f"{key}:\t{value}\n"
        copy_btn = ttk.Button(frame, text="Copy Properties")
        copy_btn.grid(row=len(properties), column=0, columnspan=2, pady=10, sticky='w', padx=5)
        copy_btn.config(command=lambda: self.copy_to_clipboard(content_to_copy, copy_btn))
        return content_to_copy

    def create_table_section_frame(self, parent, title):
        # This function now only creates and returns the LabelFrame
        return ttk.LabelFrame(parent, text=f" {title} ", padding=(10, 5))

    def populate_table_content(self, frame, title, data, columns):
        # This function populates the given frame with the table and returns content for copying
        
        # Create a container frame for the canvas and its scrollbar
        table_scroll_container = ttk.Frame(frame)
        table_scroll_container.pack(fill='both', expand=True, pady=5) # This should fill the LabelFrame

        # Create a Canvas to act as the scrollable area for the Treeview
        table_canvas = tk.Canvas(table_scroll_container, highlightthickness=0)
        table_canvas._is_table_canvas = True # Custom attribute for mousewheel handler

        # Create Treeview inside the canvas
        tree = ttk.Treeview(table_canvas, columns=columns, show='headings', height=10)
        
        # Configure columns
        total_column_width = 0
        for col in columns:
            tree.heading(col, text=col, command=lambda _col=col: self.sort_table(tree, _col, False))
            col_width = 120 # Default width for each column
            tree.column(col, minwidth=50, width=col_width, anchor='center', stretch=False)
            total_column_width += col_width
        
        for row_data in data:
            tree.insert('', 'end', values=tuple(f"{v:.4f}" if isinstance(v, float) else v for v in row_data))

        # Vertical Scrollbar for the Treeview itself (if there are more rows than height)
        vsb_tree = ttk.Scrollbar(table_scroll_container, orient="vertical", command=tree.yview)
        vsb_tree.pack(side='right', fill='y')
        tree.configure(yscrollcommand=vsb_tree.set)

        # Horizontal Scrollbar for the Canvas
        hsb_canvas = ttk.Scrollbar(table_scroll_container, orient="horizontal", command=table_canvas.xview)
        hsb_canvas.pack(side='bottom', fill='x')
        table_canvas.configure(xscrollcommand=hsb_canvas.set)

        # Place the Treeview inside the Canvas
        canvas_window_id = table_canvas.create_window((0, 0), window=tree, anchor="nw")

        # Pack the Canvas itself
        table_canvas.pack(side='left', fill='both', expand=True)

        # Function to update the canvas's scrollregion based on treeview's content width
        def update_canvas_scrollregion_and_treeview_size():
            # Get the actual width of each column after layout.
            current_total_column_width = 0
            for col_id in tree["columns"]:
                current_total_column_width += tree.column(col_id, "width")
            
            # Add a small buffer for potential internal Treeview padding/margins
            current_total_column_width += 30 
            
            # Set the canvas's scrollregion to encompass the treeview's width and height
            table_canvas.configure(scrollregion=(0, 0, current_total_column_width, tree.winfo_reqheight()))
            
            # Crucial: Update the Treeview's dimensions within the canvas.
            # This makes the Treeview itself take up the full calculated content width.
            table_canvas.itemconfig(canvas_window_id, width=current_total_column_width, height=tree.winfo_reqheight())

        # Bind to the Treeview's <Configure> event to update the canvas's scrollregion.
        # Use after_idle to ensure the Treeview has been fully rendered.
        tree.bind('<Configure>', lambda e: self.after_idle(update_canvas_scrollregion_and_treeview_size), add='+')
        # Also call it once after initial setup
        self.after_idle(update_canvas_scrollregion_and_treeview_size)

        header = "\t".join(columns)
        rows = ["\t".join(map(str, row)) for row in data]
        content_for_copy = f"--- {title} ---\n" + header + "\n" + "\n".join(rows)

        copy_btn = ttk.Button(frame, text=f"Copy {title} (TSV)")
        copy_btn.pack(pady=(10, 5), anchor='w', padx=5)
        copy_btn.config(command=lambda: self.copy_to_clipboard(content_for_copy, copy_btn))
        
        return content_for_copy

    # The create_table_section function now just acts as a wrapper to create the frame
    # and then call populate_table_content. This keeps the structure cleaner.
    def create_table_section(self, parent, title, data, columns):
        frame = self.create_table_section_frame(parent, title)
        content_for_copy = self.populate_table_content(frame, title, data, columns)
        return content_for_copy


    def sort_table(self, tree, col, reverse):
        data = [(tree.set(item, col), item) for item in tree.get_children('')]
        try:
            data.sort(key=lambda t: float(t[0]), reverse=reverse)
        except ValueError:
            data.sort(key=lambda t: t[0], reverse=reverse)
        for index, (val, item) in enumerate(data):
            tree.move(item, '', index)
        tree.heading(col, command=lambda: self.sort_table(tree, col, not reverse))

    # Modified these functions to call the new populate_table_content
    def populate_node_features_section(self, frame):
        data = self.structure_graph.x.tolist()
        node_feature_names = [
            "# Spans X", "# Spans Y", "# Spans Z",
            "X Coord", "Y Coord", "Z Coord",
            "Support", "Joint", "Nodal Mass",
            "Force X", "Force Z"
        ]
        if len(node_feature_names) == self.structure_graph.x.shape[1]:
            columns = ["Node #"] + node_feature_names
        else:
            columns = ["Node #"] + [f"F{i}" for i in range(self.structure_graph.x.shape[1])]
        table_data = [[i] + row for i, row in enumerate(data)]
        return self.populate_table_content(frame, "Node Features", table_data, columns)

    def populate_edge_features_section(self, frame):
        if not hasattr(self.structure_graph, 'edge_attr') or self.structure_graph.edge_attr is None:
            return None
        data = self.structure_graph.edge_attr.tolist()
        edge_feature_names = ["Beam", "Column", "Length"]
        if len(edge_feature_names) == self.structure_graph.edge_attr.shape[1]:
            columns = ["Edge #"] + edge_feature_names
        else:
            columns = ["Edge #"] + [f"F{i}" for i in range(self.structure_graph.edge_attr.shape[1])]
        table_data = [[i] + row for i, row in enumerate(data)]
        return self.populate_table_content(frame, "Edge Features", table_data, columns)

    def populate_edge_index_section(self, frame):
        data = self.structure_graph.edge_index.t().tolist()
        columns = ["Edge #", "Source Node", "Target Node"]
        table_data = [[i] + row for i, row in enumerate(data)]
        return self.populate_table_content(frame, "Edge Index", table_data, columns)

    def populate_targets_section(self, frame):
        if not hasattr(self.structure_graph, 'y') or self.structure_graph.y is None:
            return None
        data = self.structure_graph.y.tolist()
        num_targets = self.structure_graph.y.shape[1] if self.structure_graph.y.dim() > 1 else 1
        
        target_names = ["Disp X", "Disp Y"]
        for i in range(6):
            target_names.append(f"Moment Y {i+1}")
        for i in range(6):
            target_names.append(f"Moment Z {i+1}")
        for i in range(6):
            target_names.append(f"Shear Y {i+1}")
        for i in range(6):
            target_names.append(f"Shear Z {i+1}")

        if len(target_names) == num_targets:
            columns = ["Node #"] + target_names
        else:
            columns = ["Node #"] + [f"Target {i}" for i in range(num_targets)]
        
        if num_targets == 1:
            table_data = [[i, val] for i, val in enumerate(data)]
        else:
            table_data = [[i] + row for i, row in enumerate(data)]
        return self.populate_table_content(frame, "Target Values (y)", table_data, columns)

if __name__ == "__main__":
    root_dummy = tk.Tk()
    root_dummy.withdraw()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print("Opening file explorer to select a structure file...")
    file_path = filedialog.askopenfilename(
        initialdir=project_root,
        title="Select a PyTorch Geometric .pt file",
        filetypes=(("PyTorch Geometric files", "*.pt"), ("All files", "*.*"))
    )

    if file_path:
        try:
            structure_graph = load_structure_from_path(file_path)
            
            root = tk.Tk()
            root.withdraw()
            
            style = ttk.Style(root)
            if 'clam' in style.theme_names(): style.theme_use('clam')
            style.configure('Accent.TButton', background='#0078d4', foreground='white')
            style.map('Accent.TButton', background=[('active', '#005a9e')])
            
            dashboard = DiagnosticsDashboard(root, structure_graph, os.path.basename(file_path))
            dashboard.protocol("WM_DELETE_WINDOW", root.destroy)
            
            root.mainloop()

        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An error occurred while loading the file:\n{e}")
    else:
        print("No file selected. Exiting.")
    
    root_dummy.destroy()