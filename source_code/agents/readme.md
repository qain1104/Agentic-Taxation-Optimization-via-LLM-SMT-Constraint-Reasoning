##  System Operation Guide
In our system, we first identify the type of tax the user intends to calculate. You can initiate a conversation by selecting a tax category from the menu on the right.

###  Workflow Phases
The process is divided into three main phases:(You need to type "下一步" which means next step to enter the next phase!)

1. **Tax Variable Input**
   - Providing basic financial and tax-related data.
2. **Customized Condition Input**
   - Adjusting specific parameters or preferences based on individual scenarios.
3. **Calculation & Report Generation**
   - The system executes calculations and generates a comprehensive tax report.

---

## Test Cases
For validation purposes, we provide the following test cases.**Note: It is recommended to copy and paste the Chinese text into the Chinese version of the system for the most accurate results.**

> [!IMPORTANT]
> **Operational Note:**
> For maximum accuracy, please explicitly state the tax type you wish to calculate or select it from the list on the right side of the Gradio UI before proceeding with Phase 1 and Phase 2 inputs.


### Case #1 - Income tax
| Phase | Chinese Input (copy this) | English Translation (Reference) |
| :--- | :--- | :--- |
| **Phase 1** | 我是一名公司高管，年收400萬、太太在家當貴婦，我們有兩個小孩。 | I am a corporate executive with an annual income of 4 million. My wife is a homemaker, and we have two children. |
| **Phase 2** | 我想嘗試調整一些配置，所得總和400萬。 | I would like to try adjusting some configurations; the total income is 4 million. |


### Case #2: Cargo(Commodity) Tax

| Phase | Chinese Input (copy this) | English Translation (Reference) |
| :--- | :--- | :--- |
| **Phase 1** | 我想計算幾種車輛的貨物稅。目前 2000cc 以下的小客車每台完稅價格是 60 萬，超過 2000cc 的是 95 萬。另外，貨車和大客車每台 80 萬，機車則是每台 9 萬。 | I want to calculate the commodity tax for several types of vehicles. Currently, the tax-paid price is 600,000 for small cars (≤2000cc) and 950,000 for those >2000cc. Additionally, it's 800,000 for trucks/buses and 90,000 for motorcycles. |
| **Phase 2** | 貨車和大客車預計在 5 到 15 台之間，機車則至少要 40 台。 | Let's set some conditions: The supply of trucks/buses should be between 5 and 15 units, and motorcycles should be at least 40 units. |

---

### Case #3: Estate Tax
| Phase | Chinese Input (copy this) | English Translation (Reference) |
| :--- | :--- | :--- |
| **Phase 1** | 被繼承人於民國 111 年 8 月 8 日非因公死亡。遺產包含：土地 1,800 萬、地上物 200 萬、房屋 600 萬。銀行存款與債權共 550 萬，股票投資 400 萬，還有 120 萬的現金跟珠寶。另外，去世前 2 年內曾贈與 300 萬。家屬部分有配偶、2 名子女，以及 1 名扶養親屬，其中一人領有身心障礙手冊。 | The decedent passed away (non-duty related) on August 8, 2022. The estate includes: 18M in land, 2M in property fixtures, and 6M in housing. There is 5.5M in deposits/claims, 4M in stocks/investments, and 1.2M in cash/jewelry. Also, there was a 3M gift made within 2 years prior to death. Dependents include a spouse, 2 children, and 1 other dependent, one of whom has a disability. |
| **Phase 2** | 農業用地價值要佔土地總額的 60% 以上。未繳稅罰鍰不能超過存款債權的 10%。贈與稅跟國外扣抵額合計要達到生前贈與額的一半以上。最後，6-9 年的農業扣除額不能超過農地本身的價值。 | Agricultural land value should be at least 60% of the total land value. Unpaid tax penalties must not exceed 10% of the deposits/claims. The combined gift tax and foreign tax credits should be at least 50% of the gifts made within 2 years prior to death. Lastly, the agricultural deduction for years 6-9 must not exceed the value of the agricultural land itself. |

---

### Additional Notes
* **For Non-Chinese Speakers:**
  Please refer to `multi_agent_tax_system_en.py` and `app_gradio_en.py` to understand the implementation of our agentic service.