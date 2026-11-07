"""Export choices for an already baked character."""
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QComboBox,QCheckBox,QLabel,QDialogButtonBox


class CharacterExportDialog(QDialog):
    def __init__(self,kind,parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export Character')
        self.setMinimumWidth(max(440,self.fontMetrics().horizontalAdvance('M')*32))
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        self.format=QComboBox();self.format.addItem('FBX (.fbx)','fbx')
        if kind=='pmx':self.format.addItem('VMD (.vmd)','vmd')
        form.addRow('Format',self.format)
        self.mesh=QCheckBox('Include character mesh');self.mesh.setChecked(True)
        self.mesh.setToolTip('Checked: save the character mesh, skeleton and applied facial animation. Unchecked: save only skeleton animation; facial shape keys require the mesh.')
        layout.addWidget(self.mesh)
        self.note=QLabel();self.note.setWordWrap(True);layout.addWidget(self.note)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.format.currentIndexChanged.connect(self.update_format);self.update_format()

    def update_format(self):
        vmd=self.format.currentData()=='vmd';self.mesh.setVisible(not vmd)
        self.note.setText('Motion only, sampled at 30 fps for the same PMX model. Includes body, fingers and any applied facial morphs. Physics is not generated.' if vmd else 'Save the retargeted motion. Each take is saved as a separate file. Include the mesh to export applied facial animation.')
        self.layout().activate()
        self.fit_note()
        self.adjustSize()

    def fit_note(self):
        self.note.setMinimumHeight(self.note.heightForWidth(max(1,self.note.width())))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.fit_note()

    def options(self):
        return dict(format=self.format.currentData(),include_mesh=self.format.currentData()=='fbx' and self.mesh.isChecked())
