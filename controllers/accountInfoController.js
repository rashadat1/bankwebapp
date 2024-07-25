const axios = require('axios');
const AccountToken = require('../models/AccountToken');
require('dotenv').config();

exports.getBalance = async (req, res) => {
    const { customerID } = req.body;
    console.log('Received request to get balance for customerID: ', customerID);
    try {
        const accountTokens = await AccountToken.find({ customerID: customerID });
        if (accountTokens.length === 0) {
            return res.status(200).json({ totalBalance: 0});
        }

        const balances = await Promise.all(
            accountTokens.map(async (token) => {
                const response = await axios.get(`https://api.withmono.com/v2/accounts/${token.apiAuthToken}/balance`, {

                    headers: {
                        'Content-Type': 'application/json',
                        'accept': 'application/json',
                        'mono-sec-key': process.env.MONO_SECRET_KEY,
                    },
                });
                console.log(`Balance for account ${token.apiAuthToken}: ${response.data.data.balance}`);
                return { accountId: token.apiAuthToken, balance: parseFloat(response.data.data.balance) }
            })
        );
        // accumulator function that reduces the balances array into a single value
        const totalBalance = balances.reduce((sum, account) => sum + account.balance, 0);
        console.log('The total balance is: ', totalBalance);
        res.status(200).json({ totalBalance, balances });
        } catch(error) {
            console.error('Error fetching balance:', error);
        }
};

