/**
 * Standardised JSON response helpers.
 */

const success = (res, data, message = "Success", statusCode = 200) => {
  return res.status(statusCode).json({
    success: true,
    message,
    data,
  });
};

const error = (res, message = "Something went wrong", statusCode = 500, errors = []) => {
  return res.status(statusCode).json({
    success: false,
    message,
    errors,
    statusCode,
  });
};

const created = (res, data, message = "Created successfully") => {
  return success(res, data, message, 201);
};

const accepted = (res, data, message = "Accepted") => {
  return success(res, data, message, 202);
};

module.exports = { success, error, created, accepted };
