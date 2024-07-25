const express = require('express');
const { getBalance } = require('../controllers/accountInfoController');

const router = express.Router();

router.post('/balance', getBalance);
//router.get('transactions',transactions);

module.exports = router;