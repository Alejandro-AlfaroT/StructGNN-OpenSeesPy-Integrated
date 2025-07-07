import os
import torch
import tkinter as tk
from tkinter import ttk, messagebox
from file_explorer_tk import ask_for_file

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
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")

        # FIX: Bind mousewheel scrolling to the canvas and the inner frame
        # This makes scrolling work when the cursor is over empty space in the dashboard
        for widget in [self.canvas, self.scrollable_frame]:
            widget.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.all_diagnostics_content = []

        self.create_copy_all_button(self.scrollable_frame)
        self.all_diagnostics_content.append(self.create_properties_section(self.scrollable_frame))
        self.all_diagnostics_content.append(self.create_node_features_section(self.scrollable_frame))
        self.all_diagnostics_content.append(self.create_edge_features_section(self.scrollable_frame))
        self.all_diagnostics_content.append(self.create_edge_index_section(self.scrollable_frame))
        self.all_diagnostics_content.append(self.create_targets_section(self.scrollable_frame))
        self.finalize_setup()

    def _on_mousewheel(self, event):
        # Determine the widget under the cursor
        widget = self.winfo_containing(event.x_root, event.y_root)
        
        # If the widget is a Treeview, let it handle its own scrolling.
        # We check the widget and its parent to correctly identify the treeview.
        if isinstance(widget, ttk.Treeview) or (hasattr(widget, 'master') and isinstance(widget.master, ttk.Treeview)):
            return
        
        # Otherwise, scroll the main canvas
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_copy_all_button(self, parent):
        frame = ttk.Frame(parent, padding=(10, 10))
        frame.pack(fill='x', expand=True)
        copy_all_btn = ttk.Button(frame, text="Copy All Diagnostics to Clipboard", style="Accent.TButton")
        copy_all_btn.pack(anchor='w')
        self.copy_all_button = copy_all_btn
        frame.tkraise()

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

    def create_properties_section(self, parent):
        frame = ttk.LabelFrame(parent, text=" Graph Properties ", padding=(10, 5))
        frame.pack(fill='x', expand=True, padx=10, pady=5)
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

    def create_table_section(self, parent, title, data, columns):
        frame = ttk.LabelFrame(parent, text=f" {title} ", padding=(10, 5))
        frame.pack(fill='x', expand=True, padx=10, pady=5)
        
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill='x', expand=True, pady=5)

        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            tree.heading(col, text=col, command=lambda _col=col: self.sort_table(tree, _col, False))
            tree.column(col, width=100, anchor='center', stretch=False)
        
        for row_data in data:
            tree.insert('', 'end', values=tuple(f"{v:.4f}" if isinstance(v, float) else v for v in row_data))

        # FIX: Correctly implemented horizontal and vertical scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        tree.pack(side='left', fill='both', expand=True)

        header = "\t".join(columns)
        rows = ["\t".join(map(str, row)) for row in data]
        content_for_copy = f"--- {title} ---\n" + header + "\n" + "\n".join(rows)

        copy_btn = ttk.Button(frame, text=f"Copy {title} (TSV)")
        copy_btn.pack(pady=(10, 5), anchor='w', padx=5)
        copy_btn.config(command=lambda: self.copy_to_clipboard(content_for_copy, copy_btn))
        
        return content_for_copy

    def sort_table(self, tree, col, reverse):
        data = [(tree.set(item, col), item) for item in tree.get_children('')]
        try:
            data.sort(key=lambda t: float(t[0]), reverse=reverse)
        except ValueError:
            data.sort(reverse=reverse)
        for index, (val, item) in enumerate(data):
            tree.move(item, '', index)
        tree.heading(col, command=lambda: self.sort_table(tree, col, not reverse))

    def create_node_features_section(self, parent):
        data = self.structure_graph.x.tolist()
        columns = ["Node #"] + [f"F{i}" for i in range(self.structure_graph.x.shape[1])]
        table_data = [[i] + row for i, row in enumerate(data)]
        return self.create_table_section(parent, "Node Features", table_data, columns)

    def create_edge_features_section(self, parent):
        if not hasattr(self.structure_graph, 'edge_attr') or self.structure_graph.edge_attr is None:
            return None
        data = self.structure_graph.edge_attr.tolist()
        columns = ["Edge #"] + [f"F{i}" for i in range(self.structure_graph.edge_attr.shape[1])]
        table_data = [[i] + row for i, row in enumerate(data)]
        return self.create_table_section(parent, "Edge Features", table_data, columns)

    def create_edge_index_section(self, parent):
        data = self.structure_graph.edge_index.t().tolist()
        columns = ["Edge #", "Source Node", "Target Node"]
        table_data = [[i] + row for i, row in enumerate(data)]
        return self.create_table_section(parent, "Edge Index", table_data, columns)

    def create_targets_section(self, parent):
        if not hasattr(self.structure_graph, 'y') or self.structure_graph.y is None:
            return None
        data = self.structure_graph.y.tolist()
        num_targets = self.structure_graph.y.shape[1] if self.structure_graph.y.dim() > 1 else 1
        columns = ["Node #"] + [f"Target {i}" for i in range(num_targets)]
        
        if num_targets == 1:
            table_data = [[i, val] for i, val in enumerate(data)]
        else:
            table_data = [[i] + row for i, row in enumerate(data)]
        return self.create_table_section(parent, "Target Values (y)", table_data, columns)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print("Opening file explorer to select a structure file...")
    file_path = ask_for_file(initial_dir=project_root)

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
