from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel


class OutcomeType(str, Enum):
    BINARY = "BINARY"
    FREE_RESPONSE = "FREE_RESPONSE"
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    NUMERIC = "NUMERIC"
    PSEUDO_NUMERIC = "PSEUDO_NUMERIC"
    BOUNTIED_QUESTION = "BOUNTIED_QUESTION"
    POLL = "POLL"
    MULTI_NUMERIC = "MULTI_NUMERIC"
    DATE = "DATE"


class AddAnswersMode(str, Enum):
    ANYONE = "ANYONE"
    ONLYCREATOR = "ONLY_CREATOR"
    DISABLED = "DISABLED"


class Option(BaseModel):
    text: str
    votes: int


class Fees(BaseModel):
    platformFee: float
    liquidityFee: float
    creatorFee: float


class Fill(BaseModel):
    amount: int | float
    matchedBetId: Optional[str]
    shares: float
    timestamp: int


class Bet(BaseModel):
    contractId: str
    createdTime: int
    updatedTime: Optional[int] = None
    amount: float
    loanAmount: Optional[float] = None
    outcome: str
    shares: float
    probBefore: float
    probAfter: float
    fees: Fees
    isRedemption: bool
    replyToCommentId: Optional[str] = None
    betGroupId: Optional[str] = None
    answerId: Optional[str] = None
    limitProb: Optional[float] = None
    expiresMillisAfter: Optional[int] = None
    expiresAt: Optional[int] = None
    isApi: Optional[bool] = None
    id: Optional[str] = None
    userId: Optional[str] = None


class LiteMarket(BaseModel):
    id: str
    creatorId: str
    creatorUsername: str
    creatorName: str
    createdTime: int
    question: str
    url: str
    outcomeType: OutcomeType
    mechanism: str
    totalLiquidity: Optional[float] = None
    volume: float
    volume24Hours: float
    isResolved: bool
    uniqueBettorCount: int
    creatorAvatarUrl: Optional[str] = None
    closeTime: Optional[int] = None
    p: Optional[float] = None
    probability: Optional[float] = None
    pool: Optional[Dict[str, Any]] = None
    value: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    isLogScale: Optional[bool] = None
    resolutionTime: Optional[int] = None
    resolution: Optional[str] = None
    resolutionProbability: Optional[float] = None
    lastUpdatedTime: Optional[int] = None
    lastBetTime: Optional[int] = None
    marketTier: Optional[str] = None
    siblingContractId: Optional[str] = None
    slug: Optional[str] = None
    token: Optional[str] = None
    lastCommentTime: Optional[int] = None
    resolverId: Optional[str] = None
    comments: Optional[List[str]] = None


class FullMarket(LiteMarket):
    description: Dict[str, Any]
    textDescription: str
    answers: Optional[List[dict]] = None
    shouldAnswersSumToOne: Optional[bool] = None
    addAnswersMode: Optional[AddAnswersMode] = None
    options: Optional[List[Option]] = None
    totalBounty: Optional[float] = None
    bountyLeft: Optional[float] = None
    coverImageUrl: Optional[str] = None
    groupSlugs: List[str] = []


class User(BaseModel):
    id: str
    createdTime: int
    name: str
    username: str
    url: str
    balance: float
    totalDeposits: float
    lastBetTime: Optional[int] = None
    currentBettingStreak: Optional[int] = None
    avatarUrl: Optional[str] = None
    bio: Optional[str] = None
    bannerUrl: Optional[str] = None
    website: Optional[str] = None
    twitterHandle: Optional[str] = None
    discordHandle: Optional[str] = None
    isBot: Optional[bool] = None
    isAdmin: Optional[bool] = None
    isTrustworthy: Optional[bool] = None
    isBannedFromPosting: Optional[bool] = None
    userDeleted: Optional[bool] = None
    verifiedPhone: Optional[bool] = None
    creatorTraders: Optional[dict] = None
    signupBonusPaid: Optional[bool] = None
    isBannedFromMana: Optional[bool] = None
    nextLoanCached: Optional[int] = None
    hasSeenLoanModal: Optional[bool] = None
    isAdvancedTrader: Optional[bool] = None
    kycDocumentStatus: Optional[str] = None
    optOutBetWarnings: Optional[bool] = None
    shouldShowWelcome: Optional[bool] = None
    streakForgiveness: Optional[int] = None
    followerCountCached: Optional[int] = None
    isBannedFromSweepcash: Optional[bool] = None
    cashBalance: Optional[float] = None
    spiceBalance: Optional[float] = None
    totalCashDeposits: Optional[float] = None


class MarketPosition(BaseModel):
    userId: str
    contractId: str
    answerId: Optional[str] = None
    lastBetTime: int
    hasNoShares: bool
    hasShares: bool
    hasYesShares: bool
    invested: float
    loan: float
    id: Optional[float] = None
    maxSharesOutcome: Optional[str] = None
    totalShares: Dict[str, float]
    totalSpent: Dict[str, float]
    payout: float
    profit: float
    profitPercent: float
    from_: Optional[Dict[str, Any]] = None
    userUsername: Optional[str] = None
    userName: Optional[str] = None
    userAvatarUrl: Optional[str] = None

    def __init__(self, **kwargs):
        if "from" in kwargs:
            kwargs["from_"] = kwargs.pop("from")
        super().__init__(**kwargs)
