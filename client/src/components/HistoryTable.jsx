import "./HistoryTable.css";
import { HiTrash, HiEye } from "react-icons/hi";

const HistoryTable = ({ predictions = [], onDelete, onView }) => {
  if (!predictions.length) {
    return (
      <div className="empty-history">
        <p>No predictions yet. Start by checking a username!</p>
      </div>
    );
  }

  return (
    <div className="history-table-wrapper">
      <table className="history-table">
        <thead>
          <tr>
            <th>Username</th>
            <th>Result</th>
            <th>Confidence</th>
            <th>Risk</th>
            <th>Time</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((pred) => (
            <tr key={pred._id}>
              <td className="username-cell">@{pred.inputUsername}</td>
              <td>
                <span className={`badge ${pred.prediction === "bot" ? "badge-bot" : "badge-human"}`}>
                  {pred.prediction === "bot" ? "🤖 Bot" : "✅ Human"}
                </span>
              </td>
              <td className="mono">{(pred.confidence * 100).toFixed(1)}%</td>
              <td>
                <span className={`badge badge-${pred.riskLevel}`}>
                  {pred.riskLevel}
                </span>
              </td>
              <td className="time-cell">
                {new Date(pred.createdAt).toLocaleDateString()}
              </td>
              <td className="actions-cell">
                {onView && (
                  <button className="action-btn view-btn" onClick={() => onView(pred)} title="View">
                    <HiEye />
                  </button>
                )}
                {onDelete && (
                  <button className="action-btn delete-btn" onClick={() => onDelete(pred._id)} title="Delete">
                    <HiTrash />
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default HistoryTable;
