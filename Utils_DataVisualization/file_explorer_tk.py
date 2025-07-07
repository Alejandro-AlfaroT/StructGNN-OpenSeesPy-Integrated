import os
import tkinter as tk
from tkinter import ttk, messagebox
import re
import sys

class CustomFileExplorer(tk.Toplevel):
    """A robust file explorer using a ttk.Treeview for performance."""
    def __init__(self, parent, initial_dir, on_file_select_callback):
        super().__init__(parent)
        self.title("Select Structure File")
        self.geometry("750x600")
        self.on_file_select_callback = on_file_select_callback
        self.history = []
        self.history_pos = -1
        
        # FIX: Initialize current_path immediately to prevent AttributeError on startup search trace
        self.current_path = os.path.normpath(initial_dir)

        self.configure(bg="#f0f0f0")

        # --- Layout ---
        top_bar = ttk.Frame(self, padding=(5, 5))
        top_bar.pack(fill='x', side='top')
        self.breadcrumb_frame = ttk.Frame(self, padding=(10, 5))
        self.breadcrumb_frame.pack(fill='x', side='top')
        
        tree_frame = ttk.Frame(self)
        tree_frame.pack(expand=True, fill='both', side='top', padx=5, pady=5)
        
        search_frame = ttk.Frame(self, padding=(10, 10))
        search_frame.pack(fill='x', side='bottom')

        # --- Treeview for file listing ---
        self.tree = ttk.Treeview(tree_frame, columns=('fullpath', 'type'), show='tree', selectmode='browse')
        v_scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=v_scroll.set)
        self.tree.pack(side='left', expand=True, fill='both')
        v_scroll.pack(side='right', fill='y')

        # --- Navigation & Search Widgets ---
        self.back_btn = ttk.Button(top_bar, text="Back", command=self.go_back, state="disabled")
        self.back_btn.pack(side='left', padx=(0, 2))
        self.fwd_btn = ttk.Button(top_bar, text="Forward", command=self.go_forward, state="disabled")
        self.fwd_btn.pack(side='left')
        
        ttk.Label(search_frame, text="Search:").pack(side='left', padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.pack(fill='x', expand=True, side='left')

        # --- Bindings ---
        self.tree.bind("<Double-1>", self.on_item_double_click)
        self.search_entry.bind("<Return>", self.populate_and_filter)
        self.search_entry.bind("<Escape>", lambda e: (self.search_var.set(""), self.populate_and_filter()))
        self.search_var.trace_add("write", lambda n, i, m: self.populate_and_filter())
        self.bind("<Alt-Left>", lambda e: self.go_back())
        self.bind("<Alt-Right>", lambda e: self.go_forward())
        
        self.search_entry.bind("<Control-a>", self.select_all_text)
        self.search_entry.bind("<Control-A>", self.select_all_text)

        self.navigate(self.current_path)

    def select_all_text(self, event):
        event.widget.select_range(0, 'end')
        return 'break'

    def navigate(self, path, add_to_history=True):
        if not os.path.isdir(path):
            messagebox.showerror("Error", f"Directory not found:\n{path}", parent=self)
            return
        
        self.search_var.set("")
        self.current_path = os.path.normpath(path)
        if add_to_history:
            if self.history_pos < len(self.history) - 1: self.history = self.history[:self.history_pos + 1]
            if not self.history or self.history[-1] != self.current_path: self.history.append(self.current_path)
            self.history_pos = len(self.history) - 1
        
        self.update_nav_buttons()
        self.update_breadcrumbs()
        self.populate_and_filter()

    def update_nav_buttons(self):
        self.back_btn['state'] = 'normal' if self.history_pos > 0 else 'disabled'
        self.fwd_btn['state'] = 'normal' if self.history_pos < len(self.history) - 1 else 'disabled'

    def go_back(self):
        if self.history_pos > 0:
            self.history_pos -= 1
            self.navigate(self.history[self.history_pos], add_to_history=False)

    def go_forward(self):
        if self.history_pos < len(self.history) - 1:
            self.history_pos += 1
            self.navigate(self.history[self.history_pos], add_to_history=False)

    def update_breadcrumbs(self):
        for widget in self.breadcrumb_frame.winfo_children(): widget.destroy()
        self.update_idletasks() # Ensure frame width is calculated
        max_width = self.breadcrumb_frame.winfo_width()
        current_row = ttk.Frame(self.breadcrumb_frame); current_row.pack(fill='x', anchor='w')
        current_width = 0
        path = self.current_path
        parts = path.split(os.sep)
        built_path = ""
        for i, part in enumerate(parts):
            if not part and i > 0: continue
            built_path = os.path.join(built_path, part) if i > 0 else part + os.sep
            btn_text = part + os.sep if sys.platform == "win32" and i==0 and len(part)==2 and part.endswith(':') else part
            temp_btn = ttk.Button(current_row, text=btn_text)
            temp_sep = ttk.Label(current_row, text="/", padding=(2,0))
            temp_btn.update_idletasks(); temp_sep.update_idletasks()
            btn_width = temp_btn.winfo_reqwidth()
            sep_width = temp_sep.winfo_reqwidth() if i < len(parts) - 1 and parts[i+1] else 0
            temp_btn.destroy(); temp_sep.destroy()
            if current_width > 0 and current_width + btn_width + sep_width > max_width:
                current_row = ttk.Frame(self.breadcrumb_frame); current_row.pack(fill='x', anchor='w')
                current_width = 0
            btn = ttk.Button(current_row, text=btn_text, command=lambda p=built_path: self.navigate(p))
            btn.pack(side='left')
            current_width += btn_width
            if i < len(parts) - 1 and parts[i+1]:
                sep = ttk.Label(current_row, text="/"); sep.pack(side='left', padx=2)
                current_width += sep_width

    def populate_and_filter(self, *args):
        self.tree.delete(*self.tree.get_children())
        query = self.search_var.get().lower()
        try:
            items = sorted(os.listdir(self.current_path), key=self.natural_sort_key)
            for name in items:
                if query in name.lower():
                    full_path = os.path.join(self.current_path, name)
                    if os.path.isdir(full_path):
                        self.tree.insert('', 'end', text=f" [DIR] {name}", values=(full_path, 'dir'))
                    elif name.endswith('.pt'):
                        self.tree.insert('', 'end', text=f" [FILE] {name}", values=(full_path, 'file'))
        except OSError as e:
            self.tree.insert('', 'end', text=f"Error: {e}")

    def on_item_double_click(self, event):
        item_id = self.tree.focus()
        if not item_id: return
        values = self.tree.item(item_id, 'values')
        path, item_type = values[0], values[1]
        if item_type == 'dir':
            self.navigate(path)
        elif item_type == 'file':
            self.on_file_select_callback(path)
            # FIX: Safely destroy the window. The TclError happens when the parent is already gone.
            # This ensures we only destroy if the window still exists.
            self.destroy()

    @staticmethod
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def ask_for_file(initial_dir=None):
    """Creates a root window and launches the file explorer to get a file path."""
    root = tk.Tk()
    root.withdraw()
    selected_path = None
    def on_select(path):
        nonlocal selected_path
        selected_path = path
        root.quit() # Use quit instead of destroy to break mainloop gracefully
    explorer = CustomFileExplorer(root, initial_dir or os.getcwd(), on_select)
    def on_closing():
        nonlocal selected_path
        selected_path = None
        root.quit()
    explorer.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
    # Destroy the now-unblocked root window if it still exists
    if root.winfo_exists():
        root.destroy()
    return selected_path
