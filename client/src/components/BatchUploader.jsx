import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { HiUpload, HiDocumentText, HiX } from "react-icons/hi";
import "./BatchUploader.css";

const BatchUploader = ({ onUpload, loading = false }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState([]);

  const onDrop = useCallback((accepted) => {
    if (accepted.length === 0) return;
    const f = accepted[0];
    setFile(f);

    // Parse CSV preview (first 5 rows)
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split("\n").filter(Boolean).slice(0, 6);
      setPreview(lines);
    };
    reader.readAsText(f);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    maxSize: 5 * 1024 * 1024,
    multiple: false,
  });

  const handleSubmit = () => {
    if (file && onUpload) {
      onUpload(file);
    }
  };

  const removeFile = () => {
    setFile(null);
    setPreview([]);
  };

  return (
    <div className="batch-uploader">
      {!file ? (
        <div {...getRootProps()} className={`dropzone ${isDragActive ? "dropzone-active" : ""}`}>
          <input {...getInputProps()} />
          <HiUpload className="dropzone-icon" />
          <p className="dropzone-text">
            {isDragActive ? "Drop your CSV here..." : "Drag & drop a CSV file, or click to browse"}
          </p>
          <p className="dropzone-hint">Max 5MB · CSV format · Up to 500 usernames</p>
        </div>
      ) : (
        <div className="file-selected">
          <div className="file-info">
            <HiDocumentText className="file-icon" />
            <div>
              <p className="file-name">{file.name}</p>
              <p className="file-size">{(file.size / 1024).toFixed(1)} KB</p>
            </div>
            <button className="remove-file" onClick={removeFile}>
              <HiX />
            </button>
          </div>

          {preview.length > 0 && (
            <div className="csv-preview">
              <p className="preview-title">Preview</p>
              <div className="preview-table-wrapper">
                <table className="preview-table">
                  <thead>
                    <tr>
                      {preview[0].split(",").map((h, i) => (
                        <th key={i}>{h.trim()}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.slice(1, 6).map((row, i) => (
                      <tr key={i}>
                        {row.split(",").map((cell, j) => (
                          <td key={j}>{cell.trim()}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={loading}
            style={{ width: "100%", marginTop: 16 }}
          >
            {loading ? "Processing..." : "Start Batch Analysis"}
          </button>
        </div>
      )}
    </div>
  );
};

export default BatchUploader;
